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
    if n_neg_sample > 0:
        neg_idx = rng.choice(n_neg_total, n_neg_sample, replace=False)
        n_climber = n_climber_full[neg_idx]
        n_boulder = n_boulder_full[neg_idx]
    else:
        n_climber = np.empty(0, dtype=np.int32)
        n_boulder = np.empty(0, dtype=np.int32)
    n_label = np.zeros(n_neg_sample, dtype=np.int32)

    all_climber = np.concatenate([p_climber, n_climber])
    all_boulder = np.concatenate([p_boulder, n_boulder])
    # 4 classes: 0=ambiguous-neg (didn't ascend; could be no-try or try-fail),
    # 1=go (tried, didn't send), 2=send, 3=flash. Keep them distinct — collapsing
    # 0/1 confounds p_try with p_send and prevents alpha/pi from identifying.
    all_label = np.concatenate([p_label, n_label]).astype(np.int32)

    n_climbers = data["n_climbers"]
    n_boulders = data["n_boulders"]
    n_crags = data.get("n_crags", 0)
    n_sectors = data.get("n_sectors", 0)
    boulder_crag_idx = data["boulder_crag_idx"].numpy().astype(np.int32) if "boulder_crag_idx" in data else None
    boulder_sector_idx = data["boulder_sector_idx"].numpy().astype(np.int32) if "boulder_sector_idx" in data else None

    n_go = int((all_label == 1).sum())
    n_send = int((all_label == 2).sum())
    n_flash = int((all_label == 3).sum())
    print(f"Positives: {n_pos:,}  Negatives used: {n_neg_sample:,}  Total: {len(all_climber):,}")
    print(f"  Labels — neg={n_neg_sample:,}  go={n_go:,}  send={n_send:,}  flash={n_flash:,}")
    print(f"Climbers: {n_climbers:,}  Boulders: {n_boulders:,}  Crags: {n_crags}  Sectors: {n_sectors}")

    return {
        "climber": all_climber,
        "boulder": all_boulder,
        "label": all_label,
        "n_climbers": n_climbers,
        "n_boulders": n_boulders,
        "boulder_crag_idx": boulder_crag_idx,
        "boulder_sector_idx": boulder_sector_idx,
        "n_crags": n_crags,
        "n_sectors": n_sectors,
    }


def build_model(
    n_climbers: int,
    n_boulders: int,
    climber_data: np.ndarray,
    boulder_data: np.ndarray,
    label_data: np.ndarray,
    batch_size: int = 4096,
    boulder_group_idx: np.ndarray | None = None,
    n_groups: int = 0,
    group_name: str = "group",
    per_climber_try: bool = False,
    n_dims: int = 1,
):
    """When boulder_group_idx + n_groups are supplied, d gets a 2-level
    hierarchical prior: d_j = d_group[group(j)] + sigma_within * d_raw_j.
    Use 'sector' grouping for fine pooling, 'crag' for coarse, or None to skip.
    """
    n_total = len(climber_data)
    use_hierarchy = boulder_group_idx is not None and n_groups > 0

    with pm.Model() as model:
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=2.0)
        sigma_prolificity = pm.HalfNormal("sigma_prolificity", sigma=2.0)
        sigma_popularity = pm.HalfNormal("sigma_popularity", sigma=2.0)

        # Centered for the climber/popularity variables: with ~600k observations
        # per latent group the funnel is well-determined and centered converges
        # much faster than non-centered.
        latent_shape = n_climbers if n_dims == 1 else (n_climbers, n_dims)
        boulder_shape = n_boulders if n_dims == 1 else (n_boulders, n_dims)

        theta = pm.Normal("theta", mu=0, sigma=sigma_ability, shape=latent_shape)
        alpha = pm.Normal("alpha", mu=0, sigma=sigma_prolificity, shape=n_climbers)
        pi = pm.Normal("pi", mu=0, sigma=sigma_popularity, shape=n_boulders)

        # Two-level prior on d when a group index is supplied:
        # d_j = d_group[group(j)] + sigma_within * d_raw_j.
        # Most useful when groups are fine-grained (many groups, few obs each).
        if use_hierarchy:
            sigma_d_between = pm.HalfNormal("sigma_d_between", sigma=2.0)
            sigma_d_within = pm.HalfNormal("sigma_d_within", sigma=2.0)
            group_shape = n_groups if n_dims == 1 else (n_groups, n_dims)
            d_group_raw = pm.Normal(f"d_{group_name}_raw", mu=0, sigma=1, shape=group_shape)
            d_group = pm.Deterministic(f"d_{group_name}", sigma_d_between * d_group_raw)
            d_raw = pm.Normal("d_raw", mu=0, sigma=1, shape=boulder_shape)
            group_idx_const = pt.as_tensor_variable(boulder_group_idx.astype(np.int64))
            d = pm.Deterministic("d", d_group[group_idx_const] + sigma_d_within * d_raw)
        else:
            sigma_difficulty = pm.HalfNormal("sigma_difficulty", sigma=2.0)
            d = pm.Normal("d", mu=0, sigma=sigma_difficulty, shape=boulder_shape)

        raw_beta = pm.Normal("raw_beta", mu=0.0, sigma=1.0)
        beta = pm.Deterministic("beta", pt.softplus(raw_beta))
        # "Goldilocks" try term — climbers preferentially try problems near
        # their level. With per_climber_try, each climber gets their own
        # (gamma_i, mu_try_i) drawn from a population around the global mean.
        # Pop-level hyperparams keep things regularized.
        if per_climber_try:
            log_gamma_mu = pm.Normal("log_gamma_mu", mu=0.0, sigma=1.0)
            log_gamma_sigma = pm.HalfNormal("log_gamma_sigma", sigma=1.0)
            log_gamma = pm.Normal(
                "log_gamma", mu=log_gamma_mu, sigma=log_gamma_sigma, shape=n_climbers,
            )
            gamma_vec = pm.Deterministic("gamma_vec", pt.exp(log_gamma))

            mu_try_mu = pm.Normal("mu_try_mu", mu=0.0, sigma=1.0)
            mu_try_sigma = pm.HalfNormal("mu_try_sigma", sigma=1.0)
            mu_try_vec = pm.Normal(
                "mu_try_vec", mu=mu_try_mu, sigma=mu_try_sigma, shape=n_climbers,
            )
        else:
            gamma_scalar = pm.HalfNormal("gamma", sigma=1.0)
            mu_try_scalar = pm.Normal("mu_try", mu=0.0, sigma=1.0)

        c_mb, b_mb, l_mb = pm.Minibatch(
            climber_data, boulder_data, label_data,
            batch_size=batch_size,
        )

        theta_i = theta[c_mb]
        alpha_i = alpha[c_mb]
        d_j = d[b_mb]
        pi_j = pi[b_mb]

        if per_climber_try:
            gamma_i = gamma_vec[c_mb]
            mu_try_i = mu_try_vec[c_mb]
        else:
            gamma_i = gamma_scalar
            mu_try_i = mu_try_scalar

        diff = theta_i - d_j

        if n_dims == 1:
            try_distance = (diff - mu_try_i) ** 2
            log_p_send_g_try = -pt.softplus(-diff)
            log_p_flash_g_send = -pt.softplus(-(diff - beta))
        else:
            if per_climber_try:
                try_centered = diff - mu_try_i[:, None]
            else:
                try_centered = diff - mu_try_i
            try_distance = pt.mean(try_centered ** 2, axis=1)
            log_p_send_g_try = pt.sum(-pt.softplus(-diff), axis=1)
            log_p_flash_g_send = pt.sum(-pt.softplus(-(diff - beta)), axis=1)

        logit_try = alpha_i + pi_j - gamma_i * try_distance

        # log_sigmoid(x) = -softplus(-x), numerically stable in both tails.
        log_p_try = -pt.softplus(-logit_try)
        log_p_not_try = -pt.softplus(logit_try)

        p_send_g_try = pt.clip(pt.exp(log_p_send_g_try), 0.0, 1.0 - 1e-9)
        p_flash_g_send = pt.clip(pt.exp(log_p_flash_g_send), 0.0, 1.0 - 1e-9)
        log_p_fail_g_try = pt.log1p(-p_send_g_try)
        log_p_noflash_g_send = pt.log1p(-p_flash_g_send)

        # Per-class log-likelihoods. Class 0 (negative) is ambiguous: the climber
        # either didn't try, or tried and failed — sum those two paths via
        # logsumexp. Matches the torch loss exactly.
        log_p_try_fail = log_p_try + log_p_fail_g_try
        log_p_neg = pt.logaddexp(log_p_not_try, log_p_try_fail)
        log_p_go = log_p_try_fail
        log_p_send = log_p_try + log_p_send_g_try + log_p_noflash_g_send
        log_p_flash = log_p_try + log_p_send_g_try + log_p_flash_g_send

        ll_per_sample = pt.switch(
            pt.eq(l_mb, 0), log_p_neg,
            pt.switch(
                pt.eq(l_mb, 1), log_p_go,
                pt.switch(pt.eq(l_mb, 2), log_p_send, log_p_flash),
            ),
        )

        # Scale minibatch log-likelihood to full-dataset size so the gradient is
        # an unbiased estimate of the full-data log-likelihood.
        scale = pt.cast(n_total, "floatX") / pt.cast(l_mb.shape[0], "floatX")
        pm.Potential("loglik", ll_per_sample.sum() * scale)

    return model


def get_predictive_probs(
    theta_val, alpha_val, d_val, pi_val, beta_val,
    gamma_val=1.0, mu_try_val=0.0,
):
    from scipy.special import expit as sigmoid

    diff = np.asarray(theta_val) - np.asarray(d_val)

    try_distance = np.mean((diff - mu_try_val) ** 2)
    logit_try = alpha_val + pi_val - gamma_val * try_distance

    p_try = sigmoid(logit_try)
    p_send_given_try = np.prod(sigmoid(diff))
    p_flash_given_send = np.prod(sigmoid(diff - beta_val))

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

    def flat(name):
        return posterior[name].stack(sample=("chain", "draw")).values

    def latent_samples(name, idx):
        values = flat(name)[idx]
        if values.ndim == 1:
            return values[:, None]
        return np.moveaxis(values, -1, 0)

    theta_samples = latent_samples("theta", climber_idx)
    alpha_samples = flat("alpha")[climber_idx]
    d_samples = latent_samples("d", boulder_idx)
    pi_samples = flat("pi")[boulder_idx]
    beta_samples = flat("beta")
    if "gamma" in posterior:
        gamma_samples = flat("gamma")
    elif "gamma_vec" in posterior:
        gamma_samples = flat("gamma_vec")[climber_idx]
    else:
        gamma_samples = np.ones_like(beta_samples)
    if "mu_try" in posterior:
        mu_try_samples = flat("mu_try")
    elif "mu_try_vec" in posterior:
        mu_try_samples = flat("mu_try_vec")[climber_idx]
    else:
        mu_try_samples = np.zeros_like(beta_samples)

    n_samples = theta_samples.shape[0]

    results = {
        "p_try": [],
        "p_send_given_try": [],
        "p_flash_given_send": [],
        "p_send": [],
        "p_flash": [],
    }

    for s in range(n_samples):
        probs = get_predictive_probs(
            theta_samples[s],
            float(alpha_samples[s]),
            d_samples[s],
            float(pi_samples[s]),
            float(beta_samples[s]),
            float(gamma_samples[s]),
            float(mu_try_samples[s]),
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
    # Pool over both chain and draw dimensions, not just chain 0.
    d_flat = posterior["d"].stack(sample=("chain", "draw")).values
    d_mean = d_flat.mean(axis=-1)
    d_std = d_flat.std(axis=-1)
    return d_mean, d_std
