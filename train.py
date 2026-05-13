import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vl_predictor.dataset import build_dataset, BoulderingDataset
from vl_predictor.loss import BoulderingLoss, build_class_weights, compute_accuracy
from vl_predictor.model import BoulderingModel
from vl_predictor.validation import compute_grade_r2


def train(
    dataset_path: str = "data/dataset.pt",
    boulders_path: str = "data/boulders.jsonl",
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    save_dir: str | None = None,
    checkpoint: str | None = None,
):
    ds = BoulderingDataset(dataset_path)

    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    data = torch.load(dataset_path, weights_only=True)
    model = BoulderingModel(
        n_climbers=data["n_climbers"],
        n_boulders=data["n_boulders"],
    ).to(device)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        print(f"Loaded checkpoint from {checkpoint}")
    print(f"Running training on {device}")
    p_label = data["p_label"]
    class_weights = build_class_weights(p_label, len(data["n_label"]), device)
    loss_fn = BoulderingLoss(class_weights)

    global_params = [p for name, p in model.named_parameters() if name in ("beta", "log_gamma", "mu")]
    embedding_params = [p for name, p in model.named_parameters() if name not in ("beta", "log_gamma", "mu")]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "lr": lr, "weight_decay": weight_decay},
        {"params": global_params, "lr": lr, "weight_decay": 0.0},
    ])
    label_names = {0: "neg", 1: "fail", 2: "send", 3: "flash"}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        loss_by_label = {k: 0.0 for k in range(4)}
        count_by_label = {k: 0 for k in range(4)}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for climber, boulder, label in pbar:
            climber = climber.to(device)
            boulder = boulder.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            log_probs = model(climber, boulder)
            loss, per_sample = loss_fn(log_probs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count += 1
            total_loss += loss.item()
            correct += compute_accuracy(log_probs, label)
            total += len(label)

            for lbl, val in zip(label.tolist(), per_sample.tolist()):
                loss_by_label[lbl] += val
                count_by_label[lbl] += 1

            pbar.set_postfix(loss=f"{total_loss/batch_count:.4f}", acc=f"{correct/total:.4f}")

        acc = correct / total
        parts = []
        for k in range(4):
            if count_by_label[k] > 0:
                avg = loss_by_label[k] / count_by_label[k]
                parts.append(f"{label_names[k]}={avg:.4f}")
        r2 = compute_grade_r2(model, data, boulders_path)
        print(f"Epoch {epoch+1:3d}  loss={total_loss:.4f}  acc={acc:.4f}  {'  '.join(parts)}"
              f"  R²_diff={r2['r2_diff_only']:.4f}  R²_diff+pop={r2['r2_full']:.4f}")

        if save_dir:
            torch.save(model.state_dict(), f"{save_dir}/model-{epoch+1}.pt")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="per_crag", choices=["per_crag", "full"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-model", default=None)
    args = parser.parse_args()

    run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    build_dataset(mode=args.mode, output_path=str(run_dir / "dataset.pt"))

    model = train(
        dataset_path=str(run_dir / "dataset.pt"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=str(run_dir),
        checkpoint=args.checkpoint_model,
    )

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Model saved to {run_dir / 'model.pt'}")
