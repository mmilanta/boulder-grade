import numpy as np
import pymc as pm
import pytensor.tensor as pt
import torch


def load_and_prepare_data(
    dataset_path: str = "data/dataset.pt",
    neg_ratio: float = 3.0,
    seed: int = 42,
):
    data = torch.load(dataset_path, weights_only=True)

    p_climber = data["p_climber"].numpy().astype(np.int32)
    p_boulder = data["p_boulder"].numpy().astype(np.int32)
    p_label = data["p_label"].numpy().astype(np.int32)

    n_climber_full = data["n_climber"].numpy().astype(np.int32)
    n_boulder_full = data["n_boulder"].numpy().astype(np.int32)

    n_pos = len(p_climber)
    n_neg_total = len(n_climber_full)
    n_neg_sample = min(n_neg_total, int(n_pos * neg_ratio))

    rng = np.random.RandomState(seed)
    neg_idx = rng.choice(n_neg_total, n_neg_sample, replace=False)
    n_climber = n_climber_full[neg_idx]
    n_boulder = n_boulder_full[neg_idx]
    n_label = np.zeros(n_neg_sample, dtype=np.int32)

    all_climber = np.concatenate([p_climber, n_climber])
    all_boulder = np.concatenate([p_boulder, n_boulder])
    all_label_raw = np.concatenate([p_label, n_label])

    all_label = np.clip(all_label_raw - 1, 0, None).astype(np.int32)

    n_climbers = data["n_climbers"]
    n_boulders = data["n_boulders"]

    print(f"Positives: {n_pos:,}  Negatives (sampled): {n_neg_sample:,}  Total: {len(all_climber):,}")
    print(f"Climbers: {n_climbers:,}  Boulders: {n_boulders:,}")

    return all_climber, all_boulder, all_label, n_climbers, n_boulders


def build_model(
    n_climbers: int,
    n_boulders: int,
    climber_data: np.ndarray,
    boulder_data: np.ndarray,
    label_data: np.ndarray,
    batch_size: int = 4096,
):
    n_total = len(climber_data)

    with pm.Model() as model:
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=1.0)
        sigma_prolificity = pm.HalfNormal("sigma_prolificity", sigma=1.0)
        sigma_difficulty = pm.HalfNormal("sigma_difficulty", sigma=1.0)
        sigma_popularity = pm.HalfNormal("sigma_popularity", sigma=1.0)

        theta = pm.Normal("theta", mu=0, sigma=sigma_ability, shape=n_climbers)
        alpha = pm.Normal("alpha", mu=0, sigma=sigma_prolificity, shape=n_climbers)
        d = pm.Normal("d", mu=0, sigma=sigma_difficulty, shape=n_boulders)
        pi = pm.Normal("pi", mu=0, sigma=sigma_popularity, shape=n_boulders)

        beta = pm.Normal("beta", mu=0.0, sigma=2.0)
        c_mb, b_mb, l_mb = pm.Minibatch(
            climber_data, boulder_data, label_data,
            batch_size=batch_size,
        )

        theta_i = theta[c_mb]
        alpha_i = alpha[c_mb]
        d_j = d[b_mb]
        pi_j = pi[b_mb]

        diff = theta_i - d_j

        logit_try = alpha_i + pi_j
        logit_send = diff
        logit_flash = diff - beta

        p_try = pt.sigmoid(logit_try)
        p_send_given_try = pt.sigmoid(logit_send)
        p_flash_given_send = pt.sigmoid(logit_flash)

        p_negative = (1 - p_try) + p_try * (1 - p_send_given_try)
        p_send = p_try * p_send_given_try * (1 - p_flash_given_send)
        p_flash = p_try * p_send_given_try * p_flash_given_send

        probs = pt.stack([p_negative, p_send, p_flash], axis=1)
        probs_norm = probs / probs.sum(axis=1, keepdims=True)

        pm.Categorical(
            "obs",
            p=probs_norm,
            observed=l_mb,
            total_size=n_total,
        )

    return model


def get_predictive_probs(
    theta_val, alpha_val, d_val, pi_val, beta_val
):
    from scipy.special import expit as sigmoid

    diff = theta_val - d_val

    logit_try = alpha_val + pi_val
    logit_send = diff
    logit_flash = diff - beta_val

    p_try = sigmoid(logit_try)
    p_send_given_try = sigmoid(logit_send)
    p_flash_given_send = sigmoid(logit_flash)

    p_negative = (1 - p_try) + p_try * (1 - p_send_given_try)
    p_send = p_try * p_send_given_try * (1 - p_flash_given_send)
    p_flash = p_try * p_send_given_try * p_flash_given_send

    return {
        "p_try": p_try,
        "p_send_given_try": p_send_given_try,
        "p_flash_given_send": p_flash_given_send,
        "p_negative": p_negative,
        "p_send": p_send,
        "p_flash": p_flash,
    }


def predict_pair(
    idata,
    climber_idx: int,
    boulder_idx: int,
):
    posterior = idata.posterior

    theta_samples = posterior["theta"].values[0, :, climber_idx]
    alpha_samples = posterior["alpha"].values[0, :, climber_idx]
    d_samples = posterior["d"].values[0, :, boulder_idx]
    pi_samples = posterior["pi"].values[0, :, boulder_idx]
    beta_samples = posterior["beta"].values[0, :]

    n_samples = len(theta_samples)

    results = {
        "p_try": [],
        "p_send_given_try": [],
        "p_flash_given_send": [],
        "p_send": [],
        "p_flash": [],
    }

    for s in range(n_samples):
        probs = get_predictive_probs(
            float(theta_samples[s]),
            float(alpha_samples[s]),
            float(d_samples[s]),
            float(pi_samples[s]),
            float(beta_samples[s]),
        )
        for k in results:
            results[k].append(probs[k])

    return {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "q05": float(np.percentile(v, 5)),
            "q95": float(np.percentile(v, 95)),
        }
        for k, v in results.items()
    }


def boulder_difficulty_summary(idata):
    posterior = idata.posterior
    d_mean = posterior["d"].values[0].mean(axis=0)
    d_std = posterior["d"].values[0].std(axis=0)
    return d_mean, d_std
