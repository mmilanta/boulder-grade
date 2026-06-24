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
    return compute_grade_r2_from_arrays(
        difficulty=difficulty,
        popularity=popularity,
        boulder_to_idx=data["boulder_to_idx"],
        boulders_path=boulders_path,
    )


def compute_grade_r2_from_arrays(
    difficulty: np.ndarray,
    popularity: np.ndarray,
    boulder_to_idx: dict,
    boulders_path: str = "data/boulders.jsonl",
) -> dict:
    difficulty = np.asarray(difficulty)
    popularity = np.asarray(popularity)
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
        diffs.append(np.asarray(difficulty[i], dtype=float))
        pops.append(float(popularity[i]))
        weights.append(ascents)

    grades = np.array(grades)
    diffs = np.array(diffs)
    pops = np.array(pops)
    n = len(grades)
    if n < 3:
        return {"r2_diff_only": float("nan"), "r2_full": float("nan"), "n": n}

    weights = _normalized_grade_weights(np.array(weights, dtype=float))
    diff_features = diffs.reshape(-1, 1) if diffs.ndim == 1 else diffs
    r2_diff_only = _weighted_r2(diff_features, grades, weights)
    r2_full = _weighted_r2(np.column_stack([diff_features, pops]), grades, weights)

    return {"r2_diff_only": r2_diff_only, "r2_full": r2_full, "n": n}


def _normalized_grade_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None) ** 2
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def _weighted_r2(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")

    X_design = np.concatenate([X, np.ones((len(y), 1))], axis=1)
    sqrt_w = np.sqrt(weights)
    X_weighted = X_design * sqrt_w[:, None]
    y_weighted = y * sqrt_w
    coef = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
    pred = X_design @ coef
    residuals = y - pred
    ss_res = np.sum(weights * residuals**2)
    ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
    if not np.isfinite(ss_tot) or ss_tot <= 0:
        return float("nan")
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
