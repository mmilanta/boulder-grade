import json

import numpy as np
import torch

from vl_predictor.model import BoulderingModel


GRADE_MAPPING = {
    "2": 2, "3A": 3, "3B": 4, "3C": 5,
    "4A": 6, "4B": 7, "4C": 8,
    "5A": 9, "5B": 10, "5C": 11,
    "6A": 12, "6A+": 13, "6B": 14, "6B+": 15, "6C": 16, "6C+": 17,
    "7A": 18, "7A+": 19, "7B": 20, "7B+": 21, "7C": 22, "7C+": 23,
    "8A": 24, "8A+": 25, "8B": 26, "8B+": 27, "8C": 28, "8C+": 29,
}


def compute_grade_r2(
    model: BoulderingModel,
    data: dict,
    boulders_path: str = "data/boulders.jsonl",
) -> dict:
    model.eval()

    difficulty = model.boulder_difficulty.weight.squeeze(-1).detach().cpu().numpy()
    popularity = model.boulder_popularity.weight.squeeze(-1).detach().cpu().numpy()
    idx_to_boulder = {v: k for k, v in data["boulder_to_idx"].items()}

    boulder_meta = {}
    with open(boulders_path) as f:
        for line in f:
            b = json.loads(line)
            boulder_meta[b["boulder_id"]] = b

    grades = []
    diffs = []
    pops = []
    weights = []

    for i in range(len(difficulty)):
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
        diffs.append(float(difficulty[i]))
        pops.append(float(popularity[i]))
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


def _weighted_r2(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    X_design = np.concatenate([X, np.ones((len(y), 1))], axis=1)
    W = np.diag(weights)
    coef = np.linalg.inv(X_design.T @ W @ X_design) @ X_design.T @ W @ y
    pred = X_design @ coef
    residuals = y - pred
    ss_res = np.sum(weights * residuals**2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
    return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--boulders", default="data/boulders.jsonl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = torch.load(args.dataset, weights_only=True)
    model = BoulderingModel(
        n_climbers=data["n_climbers"],
        n_boulders=data["n_boulders"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.model, weights_only=True))

    result = compute_grade_r2(
        model=model,
        data=data,
        boulders_path=args.boulders,
    )
    print(f"n={result['n']}")
    print(f"R² (difficulty only): {result['r2_diff_only']:.4f}")
    print(f"R² (difficulty + popularity): {result['r2_full']:.4f}")
