import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vl_predictor.dataset import BoulderingDataset
from vl_predictor.model import BoulderingModel


def compute_laplace_variances(
    model: BoulderingModel,
    dataset_path: str = "data/dataset.pt",
    batch_size: int = 4096,
    weight_decay: float = 1e-4,
    n_neg_samples: int | None = None,
    device: str = "cuda",
    show_progress: bool = True,
) -> dict:
    """
    Compute diagonal Laplace posterior variances for all parameters.

    Uses the empirical Fisher (squared gradients of log-likelihood) as the
    Hessian approximation.  The prior precision from weight_decay is added
    to embedding parameters only (global scalars have weight_decay=0 in
    training).

    Returns dict with keys: climber_ability, climber_prolificity,
    boulder_difficulty, boulder_popularity, beta, log_gamma, mu.
    Each value is a 1-D tensor of variances (standard deviations = sqrt(var)).
    """
    data = torch.load(dataset_path, weights_only=True)
    model = model.to(device)
    model.eval()

    n_climbers = model.climber_ability.num_embeddings
    n_boulders = model.boulder_difficulty.num_embeddings

    beta_val = model.beta.detach()
    log_gamma_val = model.log_gamma.detach()
    mu_val = model.mu.detach()
    gamma_val = torch.exp(log_gamma_val)

    f_ability = torch.zeros(n_climbers, device=device)
    f_prolificity = torch.zeros(n_climbers, device=device)
    f_difficulty = torch.zeros(n_boulders, device=device)
    f_popularity = torch.zeros(n_boulders, device=device)
    f_beta = torch.zeros(1, device=device)
    f_log_gamma = torch.zeros(1, device=device)
    f_mu = torch.zeros(1, device=device)

    p_climber = data["p_climber"].to(device)
    p_boulder = data["p_boulder"].to(device)
    p_label = data["p_label"].to(device)

    n_climber = data["n_climber"].to(device)
    n_boulder = data["n_boulder"].to(device)
    n_label = data["n_label"].to(device)

    if n_neg_samples is not None and n_neg_samples < len(n_climber):
        idx = torch.randperm(len(n_climber), device=device)[:n_neg_samples]
        n_climber = n_climber[idx]
        n_boulder = n_boulder[idx]
        n_label = n_label[idx]

    all_climber = torch.cat([p_climber, n_climber])
    all_boulder = torch.cat([p_boulder, n_boulder])
    all_label = torch.cat([p_label, n_label])
    n_total = len(all_climber)

    pbar = tqdm(
        range(0, n_total, batch_size),
        desc="Fisher diagonal",
        disable=not show_progress,
    )
    for start in pbar:
        end = min(start + batch_size, n_total)
        c_idx = all_climber[start:end]
        b_idx = all_boulder[start:end]
        labels = all_label[start:end]
        model_labels = (labels - 1).clamp(min=0)

        theta = model.climber_ability(c_idx).squeeze(-1).detach()
        alpha = model.climber_prolificity(c_idx).squeeze(-1).detach()
        d = model.boulder_difficulty(b_idx).squeeze(-1).detach()
        pi = model.boulder_popularity(b_idx).squeeze(-1).detach()

        grads_sq = _fisher_contributions(
            theta, alpha, d, pi,
            beta_val, gamma_val, log_gamma_val, mu_val,
            model_labels,
        )

        f_ability.index_add_(0, c_idx, grads_sq["theta"])
        f_prolificity.index_add_(0, c_idx, grads_sq["alpha"])
        f_difficulty.index_add_(0, b_idx, grads_sq["d"])
        f_popularity.index_add_(0, b_idx, grads_sq["pi"])
        f_beta += grads_sq["beta"].sum()
        f_log_gamma += grads_sq["log_gamma"].sum()
        f_mu += grads_sq["mu"].sum()

    prior_prec = 2.0 * weight_decay

    result = {
        "climber_ability": (1.0 / (f_ability + prior_prec + 1e-12)).cpu(),
        "climber_prolificity": (1.0 / (f_prolificity + prior_prec + 1e-12)).cpu(),
        "boulder_difficulty": (1.0 / (f_difficulty + prior_prec + 1e-12)).cpu(),
        "boulder_popularity": (1.0 / (f_popularity + prior_prec + 1e-12)).cpu(),
        "beta": (1.0 / (f_beta + 1e-12)).cpu(),
        "log_gamma": (1.0 / (f_log_gamma + 1e-12)).cpu(),
        "mu": (1.0 / (f_mu + 1e-12)).cpu(),
    }
    return result


def _fisher_contributions(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    pi: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    log_gamma: torch.Tensor,
    mu: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Per-example squared gradients of log p(label | params) w.r.t. each
    scalar parameter.  All inputs are (batch,) tensors.  labels are
    model labels: 0=negative, 1=send, 2=flash.

    Returns dict mapping parameter name to (batch,) tensor of squared gradients.
    """
    delta = theta - d
    delta_mu = delta - mu

    a_t = alpha + pi - gamma * delta_mu ** 2
    a_s = delta
    a_f = delta - beta

    p_t = torch.sigmoid(a_t)
    p_s = torch.sigmoid(a_s)
    p_f = torch.sigmoid(a_f)

    dp_t = p_t * (1.0 - p_t)
    dp_s = p_s * (1.0 - p_s)
    dp_f = p_f * (1.0 - p_f)

    p_neg = 1.0 - p_t * p_s
    p_send = p_t * p_s * (1.0 - p_f)
    p_flash = p_t * p_s * p_f

    two_g_dm = 2.0 * gamma * delta_mu
    common_coeff = p_s * dp_t
    send_coeff = p_t * dp_s
    flash_coeff = p_t * p_s * dp_f

    shared_theta = common_coeff * (-two_g_dm) + send_coeff * 1.0
    a_f_coeff_theta = flash_coeff * 1.0

    shared_d = common_coeff * two_g_dm + send_coeff * (-1.0)
    a_f_coeff_d = flash_coeff * (-1.0)

    shared_alpha = common_coeff * 1.0
    a_f_coeff_alpha = torch.zeros_like(common_coeff)

    shared_pi = common_coeff * 1.0
    a_f_coeff_pi = torch.zeros_like(common_coeff)

    shared_beta = torch.zeros_like(common_coeff)
    a_f_coeff_beta = flash_coeff * (-1.0)

    shared_mu = common_coeff * two_g_dm
    a_f_coeff_mu = torch.zeros_like(common_coeff)

    shared_log_gamma = common_coeff * (-gamma * delta_mu ** 2)
    a_f_coeff_log_gamma = torch.zeros_like(common_coeff)

    is_neg = (labels == 0).float()
    is_send = (labels == 1).float()
    is_flash = (labels == 2).float()

    coef_shared = is_neg * (-1.0) + is_send * (1.0 - p_f) + is_flash * p_f
    coef_flash = is_neg * 0.0 + is_send * (-1.0) + is_flash * 1.0

    dp_theta = coef_shared * shared_theta + coef_flash * a_f_coeff_theta
    dp_d = coef_shared * shared_d + coef_flash * a_f_coeff_d
    dp_alpha = coef_shared * shared_alpha + coef_flash * a_f_coeff_alpha
    dp_pi = coef_shared * shared_pi + coef_flash * a_f_coeff_pi
    dp_beta = coef_shared * shared_beta + coef_flash * a_f_coeff_beta
    dp_mu = coef_shared * shared_mu + coef_flash * a_f_coeff_mu
    dp_log_gamma = coef_shared * shared_log_gamma + coef_flash * a_f_coeff_log_gamma

    p_k = torch.where(labels == 0, p_neg, torch.where(labels == 1, p_send, p_flash))
    inv_p_k = 1.0 / p_k.clamp(min=1e-12)

    return {
        "theta": (inv_p_k * dp_theta) ** 2,
        "alpha": (inv_p_k * dp_alpha) ** 2,
        "d": (inv_p_k * dp_d) ** 2,
        "pi": (inv_p_k * dp_pi) ** 2,
        "beta": (inv_p_k * dp_beta) ** 2,
        "log_gamma": (inv_p_k * dp_log_gamma) ** 2,
        "mu": (inv_p_k * dp_mu) ** 2,
    }


def predictive_uncertainty(
    model: BoulderingModel,
    variances: dict,
    climber_idx: int,
    boulder_idx: int,
    n_samples: int = 500,
) -> dict:
    """
    Monte Carlo estimate of predictive uncertainty for a specific
    (climber, boulder) pair.  Samples from the Laplace posterior over
    each parameter and returns mean ± std for each probability.
    """
    device = next(model.parameters()).device
    model.eval()

    def _var(name: str, idx: int | None = None) -> float:
        v = variances[name].detach()
        if idx is not None:
            v = v[idx]
        return float(v)

    var_theta = _var("climber_ability", climber_idx)
    var_alpha = _var("climber_prolificity", climber_idx)
    var_d = _var("boulder_difficulty", boulder_idx)
    var_pi = _var("boulder_popularity", boulder_idx)
    var_beta = _var("beta")
    var_log_gamma = _var("log_gamma")
    var_mu = _var("mu")

    theta_map = float(model.climber_ability.weight[climber_idx, 0].detach())
    alpha_map = float(model.climber_prolificity.weight[climber_idx, 0].detach())
    d_map = float(model.boulder_difficulty.weight[boulder_idx, 0].detach())
    pi_map = float(model.boulder_popularity.weight[boulder_idx, 0].detach())
    beta_map = float(model.beta.detach())
    log_gamma_map = float(model.log_gamma.detach())
    mu_map = float(model.mu.detach())

    VARIANCE_THRESHOLD = 1e6  # treat parameters with var > threshold as unidentified

    generator = torch.Generator(device=device).manual_seed(42)

    def _sample(val, var):
        if var > VARIANCE_THRESHOLD:
            return torch.full((n_samples,), val, device=device)
        return val + torch.randn(n_samples, generator=generator, device=device) * (var ** 0.5 + 1e-12)

    theta_samples = _sample(theta_map, var_theta)
    alpha_samples = _sample(alpha_map, var_alpha)
    d_samples = _sample(d_map, var_d)
    pi_samples = _sample(pi_map, var_pi)
    beta_samples = _sample(beta_map, var_beta)
    log_gamma_samples = _sample(log_gamma_map, var_log_gamma)
    mu_samples = _sample(mu_map, var_mu)

    with torch.no_grad():
        delta_samples = theta_samples - d_samples
        gamma_samples = torch.exp(log_gamma_samples)

        logit_try = alpha_samples + pi_samples - gamma_samples * (delta_samples - mu_samples) ** 2
        logit_send = delta_samples
        logit_flash = delta_samples - beta_samples

        p_try = torch.sigmoid(logit_try)
        p_send_given_try = torch.sigmoid(logit_send)
        p_flash_given_send = torch.sigmoid(logit_flash)

        p_send = p_try * p_send_given_try * (1.0 - p_flash_given_send)
        p_flash = p_try * p_send_given_try * p_flash_given_send

    def _stats(t: torch.Tensor):
        t = t.cpu()
        return {"mean": float(t.mean()), "std": float(t.std())}

    return {
        "p_try": _stats(p_try),
        "p_send_given_try": _stats(p_send_given_try),
        "p_flash_given_send": _stats(p_flash_given_send),
        "p_send": _stats(p_send),
        "p_flash": _stats(p_flash),
    }


def delta_predictive_std(
    model: BoulderingModel,
    variances: dict,
    climber_idx: int,
    boulder_idx: int,
) -> dict:
    """
    Fast delta-method approximation of predictive standard deviation
    for P(send|try) and P(flash|send).  No sampling — pure arithmetic.

    Var(P_s) ≈ [p_s·(1-p_s)]² · (Var(θ) + Var(d))

    Returns dict with keys: p_send_given_try_std, p_flash_given_send_std,
    diff (θ-d), p_send_given_try, p_flash_given_send.
    """
    import math

    theta = float(model.climber_ability.weight[climber_idx, 0].detach())
    d = float(model.boulder_difficulty.weight[boulder_idx, 0].detach())
    beta_val = float(model.beta.detach())

    var_theta = float(variances["climber_ability"][climber_idx].detach())
    var_d = float(variances["boulder_difficulty"][boulder_idx].detach())
    var_beta = float(variances["beta"].detach())

    diff = theta - d
    p_s = 1.0 / (1.0 + math.exp(-diff))
    p_f = 1.0 / (1.0 + math.exp(-(diff - beta_val)))

    dp_s = p_s * (1.0 - p_s)
    dp_f = p_f * (1.0 - p_f)

    std_send = abs(dp_s) * (var_theta + var_d) ** 0.5
    std_flash = abs(dp_f) * (var_theta + var_d + var_beta) ** 0.5

    return {
        "diff": diff,
        "p_send_given_try": p_s,
        "p_send_given_try_std": std_send,
        "p_flash_given_send": p_f,
        "p_flash_given_send_std": std_flash,
    }


def save_variances(variances: dict, path: str) -> None:
    torch.save(variances, path)


def load_variances(path: str) -> dict:
    return torch.load(path, weights_only=True)


def verify_gradients(model: BoulderingModel, dataset_path: str, device: str = "cuda"):
    """
    Compare analytical Fisher gradients against autograd for a small random
    batch.  Prints max absolute error per parameter.
    """
    from torch.func import vmap, grad

    data = torch.load(dataset_path, weights_only=True)
    model = model.to(device)
    model.eval()

    idx = torch.randperm(len(data["p_climber"]))[:64]
    c_idx = data["p_climber"][idx].to(device)
    b_idx = data["p_boulder"][idx].to(device)
    labels = data["p_label"][idx].to(device)
    model_labels = (labels - 1).clamp(min=0)

    theta = model.climber_ability(c_idx).squeeze(-1).detach().requires_grad_()
    alpha = model.climber_prolificity(c_idx).squeeze(-1).detach().requires_grad_()
    d = model.boulder_difficulty(b_idx).squeeze(-1).detach().requires_grad_()
    pi = model.boulder_popularity(b_idx).squeeze(-1).detach().requires_grad_()
    beta = model.beta.detach().requires_grad_()
    log_gamma = model.log_gamma.detach().requires_grad_()
    mu = model.mu.detach().requires_grad_()

    def _log_prob(th, al, dd, pp, bt, lg, mm, lbl):
        delta = th - dd
        gamma = torch.exp(lg)
        dm = delta - mm
        a_t = al + pp - gamma * dm ** 2
        a_s = delta
        a_f = delta - bt
        p_t = torch.sigmoid(a_t)
        p_s = torch.sigmoid(a_s)
        p_f = torch.sigmoid(a_f)
        p_neg = (1.0 - p_t * p_s).clamp(min=1e-8)
        p_send = (p_t * p_s * (1.0 - p_f)).clamp(min=1e-8)
        p_flash = (p_t * p_s * p_f).clamp(min=1e-8)
        return torch.where(
            lbl == 0,
            torch.log(p_neg),
            torch.where(lbl == 1, torch.log(p_send), torch.log(p_flash)),
        )

    batch_grads = vmap(
        grad(_log_prob, argnums=(0, 1, 2, 3, 4, 5, 6)),
        in_dims=(0, 0, 0, 0, None, None, None, 0),
    )(theta, alpha, d, pi, beta, log_gamma, mu, model_labels)

    gr_theta, gr_alpha, gr_d, gr_pi, gr_beta, gr_log_gamma, gr_mu = batch_grads

    gamma_val = torch.exp(model.log_gamma.detach())
    analytic = _fisher_contributions(
        theta.detach(), alpha.detach(), d.detach(), pi.detach(),
        model.beta.detach(), gamma_val, model.log_gamma.detach(), model.mu.detach(),
        model_labels,
    )

    names = ["theta", "alpha", "d", "pi", "beta", "log_gamma", "mu"]
    autograd_grads = [gr_theta, gr_alpha, gr_d, gr_pi, gr_beta, gr_log_gamma, gr_mu]

    print("Gradient verification (analytical vs autograd):")
    max_err = 0.0
    for name, ag in zip(names, autograd_grads):
        an = analytic[name].sqrt()
        err = (ag.abs() - an).abs().max().item()
        max_err = max(max_err, err)
        print(f"  {name:>12s}: max |error| = {err:.2e}")
    if max_err < 1e-4:
        print("  ✓ All gradients match within tolerance.")
    else:
        print("  ⚠ Some gradients differ — check numerical stability.")
    return max_err < 1e-4


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Laplace uncertainty estimation")
    parser.add_argument("--model", default="data/model.pt")
    parser.add_argument("--dataset", default="data/dataset.pt")
    parser.add_argument("--boulders", default="data/boulders.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-neg-samples", type=int, default=None,
                        help="Max negative samples to process (default: all)")
    parser.add_argument("--output", default=None,
                        help="Path to save variances (default: alongside model)")
    parser.add_argument("--verify", action="store_true",
                        help="Run gradient verification before computing")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    data = torch.load(args.dataset, weights_only=True)
    model = BoulderingModel(
        n_climbers=data["n_climbers"],
        n_boulders=data["n_boulders"],
    ).to(args.device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    print(f"Loaded model from {args.model}")

    if args.verify:
        ok = verify_gradients(model, args.dataset, args.device)
        if not ok:
            print("Gradient verification failed. Aborting.")
            raise SystemExit(1)

    variances = compute_laplace_variances(
        model=model,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        n_neg_samples=args.n_neg_samples,
        device=args.device,
    )

    if args.output is None:
        model_path = Path(args.model)
        args.output = str(model_path.with_suffix("")) + "_variances.pt"

    save_variances(variances, args.output)
    print(f"\nVariances saved to {args.output}")
    print("Parameter uncertainties (±1 std):")
    print(f"{'Parameter':>22s} {'mean std':>10s} {'median std':>10s} {'min std':>10s} {'max std':>10s} {'status'}")
    print("-" * 72)
    IDENTIFIED_THRESHOLD = 100.0

    param_list = [
        ("climber_ability", variances["climber_ability"]),
        ("climber_prolificity", variances["climber_prolificity"]),
        ("boulder_difficulty", variances["boulder_difficulty"]),
        ("boulder_popularity", variances["boulder_popularity"]),
        ("beta", variances["beta"]),
        ("log_gamma", variances["log_gamma"]),
        ("mu", variances["mu"]),
    ]
    for name, v in param_list:
        s = v.sqrt()
        status = "identified" if s.median().item() < IDENTIFIED_THRESHOLD else "UNIDENTIFIED"
        print(f"  {name:>20s}: {s.mean().item():>10.4f} {s.median().item():>10.4f} "
              f"{s.min().item():>10.4f} {s.max().item():>10.4f}  {status}")
