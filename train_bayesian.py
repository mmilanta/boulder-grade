import argparse
import json
from datetime import datetime
from pathlib import Path

import pymc as pm
import torch

from vl_predictor.bayesian_model import build_model, load_and_prepare_data, boulder_difficulty_summary
from vl_predictor.dataset import build_dataset
from vl_predictor.validation import compute_grade_r2_from_arrays


def compute_bayesian_grade_r2(
    idata,
    boulders_path: str,
    boulder_to_idx: dict,
):
    d_mean, _ = boulder_difficulty_summary(idata)

    posterior = idata.posterior
    pi_mean = posterior["pi"].stack(sample=("chain", "draw")).values.mean(axis=-1)

    return compute_grade_r2_from_arrays(
        difficulty=d_mean,
        popularity=pi_mean,
        boulder_to_idx=boulder_to_idx,
        boulders_path=boulders_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian bouldering model with PyMC")
    parser.add_argument("--mode", default="per_crag", choices=["per_crag", "full"])
    parser.add_argument("--method", default="advi",
                        choices=["advi", "fullrank_advi", "nuts"])
    parser.add_argument("--n-iter", type=int, default=50000,
                        help="Number of iterations (ADVI / fullrank)")
    parser.add_argument("--draws", type=int, default=2000,
                        help="Posterior draws (NUTS) or samples from approx")
    parser.add_argument("--tune", type=int, default=2000,
                        help="NUTS tuning steps")
    parser.add_argument("--chains", type=int, default=4,
                        help="NUTS chains")
    parser.add_argument("--cores", type=int, default=4,
                        help="NUTS cores")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--neg-ratio", type=float, default=3.0,
                        help="Ratio of negative to positive samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for negative sampling and inference")
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="During ADVI, sample posterior + log grade-R² + checkpoint every N iter (0=off)")
    parser.add_argument("--hierarchy", default="none", choices=["none", "crag", "sector"],
                        help="2-level hierarchical prior on d, pooling boulders within this group")
    parser.add_argument("--per-climber-try", action="store_true",
                        help="Per-climber Goldilocks try curve (gamma_i, mu_try_i)")
    parser.add_argument("--n-dims", type=int, default=1,
                        help="Number of latent ability/difficulty dimensions")
    parser.add_argument("--boulders", default="data/boulders.jsonl")
    args = parser.parse_args()
    if args.n_dims < 1:
        raise ValueError("--n-dims must be >= 1")
    if args.neg_ratio < 0:
        raise ValueError("--neg-ratio must be >= 0")

    run_dir = Path("runs") / ("bayesian_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    dataset_path = str(run_dir / "dataset.pt")
    build_dataset(
        mode=args.mode,
        output_path=dataset_path,
        negative_sample_ratio=args.neg_ratio,
        seed=args.seed,
    )

    prep = load_and_prepare_data(
        dataset_path, neg_ratio=args.neg_ratio, seed=args.seed,
    )
    print(f"Data: {len(prep['climber']):,} samples")

    if args.hierarchy == "crag":
        group_idx, n_groups, group_name = prep["boulder_crag_idx"], prep["n_crags"], "crag"
    elif args.hierarchy == "sector":
        group_idx, n_groups, group_name = prep["boulder_sector_idx"], prep["n_sectors"], "sector"
    else:
        group_idx, n_groups, group_name = None, 0, "group"

    model = build_model(
        n_climbers=prep["n_climbers"],
        n_boulders=prep["n_boulders"],
        climber_data=prep["climber"],
        boulder_data=prep["boulder"],
        label_data=prep["label"],
        batch_size=args.batch_size,
        boulder_group_idx=group_idx,
        n_groups=n_groups,
        group_name=group_name,
        per_climber_try=args.per_climber_try,
        n_dims=args.n_dims,
    )

    data = torch.load(dataset_path, weights_only=True)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    val_log_path = run_dir / "validation_log.jsonl"

    def _log_r2(approx, losses, i):  # pymc callback signature: (approx, losses, i)
        del losses
        if args.eval_every <= 0 or i == 0 or i % args.eval_every != 0:
            return
        try:
            tmp_idata = approx.sample(draws=args.draws)
            r2 = compute_bayesian_grade_r2(tmp_idata, args.boulders, data["boulder_to_idx"])

            ckpt_path = checkpoints_dir / f"posterior_iter{i:07d}.nc"
            tmp_idata.to_netcdf(str(ckpt_path))

            with open(val_log_path, "a") as f:
                f.write(json.dumps({"iter": i, **r2, "checkpoint": ckpt_path.name}) + "\n")

            print(f"  [iter {i:>7d}] R²_diff={r2['r2_diff_only']:.4f}  "
                  f"R²_diff+pop={r2['r2_full']:.4f}  n={r2['n']}  → {ckpt_path.name}")
        except Exception as e:
            print(f"  [iter {i:>7d}] R² eval failed: {e}")

    with model:
        if args.method == "nuts":
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                random_seed=args.seed,
            )
        elif args.method in ("advi", "fullrank_advi"):
            callbacks = [
                pm.callbacks.CheckParametersConvergence(
                    diff="absolute", tolerance=1e-3
                ),
            ]
            if args.eval_every > 0:
                callbacks.append(_log_r2)
            approx = pm.fit(
                n=args.n_iter,
                method=args.method,
                callbacks=callbacks,
                progressbar=True,
                obj_optimizer=pm.adam(learning_rate=args.lr),
                random_seed=args.seed,
            )
            idata = approx.sample(draws=args.draws)

    idata_path = run_dir / "posterior.nc"
    idata.to_netcdf(str(idata_path))
    print(f"Posterior samples saved to {idata_path}")

    r2 = compute_bayesian_grade_r2(idata, args.boulders, data["boulder_to_idx"])
    print(f"n={r2['n']}")
    print(f"R² (difficulty only): {r2['r2_diff_only']:.4f}")
    print(f"R² (difficulty + popularity): {r2['r2_full']:.4f}")

    with open(run_dir / "validation.json", "w") as f:
        json.dump(r2, f, indent=2)
