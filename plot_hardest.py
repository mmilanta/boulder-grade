import argparse
import json
import sys

import matplotlib.pyplot as plt
import torch

from vl_predictor.model import BoulderingModel


def plot_hardest(model_path: str, dataset_path: str, boulders_path: str, top_n: int = 30):
    data = torch.load(dataset_path, weights_only=True)
    model = BoulderingModel(
        n_climbers=data["n_climbers"],
        n_boulders=data["n_boulders"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    difficulty = model.boulder_difficulty.weight.squeeze(-1).detach().cpu()

    boulder_by_idx = {}
    idx_to_boulder = data["boulder_to_idx"]
    for bid, i in idx_to_boulder.items():
        boulder_by_idx[i] = bid

    bid_to_meta = {}
    with open(boulders_path) as f:
        for line in f:
            b = json.loads(line)
            bid_to_meta[b["boulder_id"]] = b

    entries = []
    for i in range(len(difficulty)):
        bid = boulder_by_idx.get(i)
        if bid is None:
            continue
        meta = bid_to_meta.get(bid, {})
        name = meta.get("boulder_name", f"boulder_{bid}")
        crag = meta.get("crag_name", "?")
        grade = meta.get("grade_community", "?")
        entries.append((difficulty[i].item(), name, crag, grade, bid))

    entries.sort(key=lambda x: x[0], reverse=True)
    entries = entries[:top_n]

    diffs = [e[0] for e in entries]
    labels = [f"{e[1]}  [{e[3]}]  {e[2]}" for e in entries]

    fig, ax = plt.subplots(figsize=(10, 0.5 * top_n + 1))
    colors = ["#d62728" if d > 2.0 else "#1f77b4" for d in diffs]
    bars = ax.barh(range(len(labels)), diffs, color=colors)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Inferred difficulty")
    ax.set_title(f"Top {top_n} Hardest Boulders")

    for bar, d in zip(bars, diffs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{d:.2f}", va="center", fontsize=7)

    fig.tight_layout()
    out = "hardest.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/latest/model.pt")
    parser.add_argument("--dataset", default="data/dataset.pt")
    parser.add_argument("--boulders", default="data/boulders.jsonl")
    parser.add_argument("-n", "--top-n", type=int, default=30)
    args = parser.parse_args()
    plot_hardest(args.model, args.dataset, args.boulders, args.top_n)
