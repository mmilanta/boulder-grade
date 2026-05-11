import torch


class BoulderingLoss:
    """4-class NLL with label-0 ambiguity.

    For label=0 (negative), we don't know if the climber tried or not,
    so the likelihood is p_not_try + p_try_fail summed via logsumexp.
    """

    def __init__(self, class_weights: torch.Tensor):
        self.class_weights = class_weights

    def __call__(self, log_probs: torch.Tensor, label: torch.Tensor):
        log_prob_neg = torch.logsumexp(log_probs[:, [0, 1]], dim=1)

        per_sample = torch.empty(len(label), device=log_probs.device)
        for k in range(4):
            mask = (label == k)
            if mask.any():
                if k == 0:
                    per_sample[mask] = -log_prob_neg[mask]
                else:
                    per_sample[mask] = -log_probs[mask, k]

        weighted = per_sample * self.class_weights[label]
        return weighted.mean(), per_sample


def compute_accuracy(log_probs: torch.Tensor, label: torch.Tensor) -> int:
    """Accuracy aware of label-0 ambiguity.

    For label=0, both pred=0 and pred=1 count as correct since we don't
    have ground truth to distinguish 'didn't try' from 'tried and failed'.
    """
    pred = log_probs.argmax(dim=-1)
    correct = 0
    for k in range(4):
        mask = (label == k)
        if mask.any():
            if k == 0:
                correct += ((pred[mask] == 0) | (pred[mask] == 1)).sum().item()
            else:
                correct += (pred[mask] == k).sum().item()
    return correct


def build_class_weights(
    p_label: torch.Tensor,
    n_label_count: int,
    device: torch.device,
) -> torch.Tensor:
    class_counts = torch.zeros(4)
    class_counts[0] = n_label_count
    class_counts[1] = (p_label == 1).sum().item()
    class_counts[2] = (p_label == 2).sum().item()
    class_counts[3] = (p_label == 3).sum().item()
    weights = (1.0 / class_counts).to(device)
    return weights / weights.sum() * 4
