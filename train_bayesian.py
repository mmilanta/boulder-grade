import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pymc as pm
import torch

from vl_predictor.bayesian_model import build_model, load_and_prepare_data, boulder_difficulty_summary
from vl_predictor.dataset import build_dataset
from vl_predictor.validation import GRADE_MAPPING, _weighted_r2


def compute_bayesian_grade_r2(
    idata,
    boulders_path: str,
    boulder_to_idx: dict,
):
    d_mean, d_std = boulder_difficulty_summary(idata)

    posterior = idata.posterior
    pi_mean = posterior["pi"].values[0].mean(axis=0)

    idx_to_boulder = {v: k for k, v in boulder_to_idx.items()}

    boulder_meta = {}
    with open(boulders_path) as f:
        for line in f:
            b = json.loads(line)
            boulder_meta[b["boulder_id"]] = b

    grades = []
    diffs = []
    pops = []
    weights = []

    for i in range(len(d_mean)):
        bid = idx_to_boulder.get(i)
        if bid is None:
            continue
        meta = boulder_meta.get(bid)
        if meta is None:
            continue
        grade_num = GRADE_MAPPING.get(meta.get("grade_community", ""))
        if grade_num is None:
            continue
        ascents = meta.get("ascent_count", 0)
        grades.append(grade_num)
        diffs.append(float(d_mean[i]))
        pops.append(float(pi_mean[i]))
        weights.append(ascents)

    grades = np.array(grades)
    diffs = np.array(diffs)
    pops = np.array(pops)
    weights = np.array(weights, dtype=float) ** 2
    weights = weights / weights.sum()

    n = len(grades)
    if n < 3:
        return {"r2_diff_only": float("nan"), "r2_full": float("nan"), "n": n}

    r2_diff_only = _weighted_r2(diffs.reshape(-1, 1), grades, weights)
    r2_full = _weighted_r2(np.stack([diffs, pops], axis=1), grades, weights)

    return {"r2_diff_only": r2_diff_only, "r2_full": r2_full, "n": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bayesian bouldering model with PyMC")
    parser.add_argument("--mode", default="per_crag", choices=["per_crag", "full"])
    parser.add_argument("--n-iter", type=int, default=50000,
                        help="Number of ADVI iterations")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--neg-ratio", type=float, default=3.0,
                        help="Ratio of negative to positive samples")
    parser.add_argument("--dataset", default="data/dataset.pt")
    parser.add_argument("--boulders", default="data/boulders.jsonl")
    args = parser.parse_args()

    run_dir = Path("runs") / ("bayesian_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    build_dataset(mode=args.mode, output_path=args.dataset)
    shutil.copy2(args.dataset, run_dir / "dataset.pt")

    all_climber, all_boulder, all_label, n_climbers, n_boulders = load_and_prepare_data(
        args.dataset, neg_ratio=args.neg_ratio
    )
    print(f"Data: {len(all_climber):,} samples")

    model = build_model(
        n_climbers=n_climbers,
        n_boulders=n_boulders,
        climber_data=all_climber,
        boulder_data=all_boulder,
        label_data=all_label,
        batch_size=args.batch_size,
    )

    with model:
        approx = pm.fit(
            n=args.n_iter,
            method="advi",
            callbacks=[
                pm.callbacks.CheckParametersConvergence(
                    diff="absolute", tolerance=1e-3
                ),
            ],
            progressbar=True,
            obj_optimizer=pm.adam(learning_rate=args.lr),
        )

    idata = approx.sample(draws=500)
    idata_path = run_dir / "posterior.nc"
    idata.to_netcdf(str(idata_path))
    print(f"Posterior samples saved to {idata_path}")

    data = torch.load(args.dataset, weights_only=True)
    r2 = compute_bayesian_grade_r2(idata, args.boulders, data["boulder_to_idx"])
    print(f"n={r2['n']}")
    print(f"R² (difficulty only): {r2['r2_diff_only']:.4f}")
    print(f"R² (difficulty + popularity): {r2['r2_full']:.4f}")

    with open(run_dir / "validation.json", "w") as f:
        json.dump(r2, f, indent=2)
