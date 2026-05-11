import torch
import torch.nn as nn


class BoulderingModel(nn.Module):
    def __init__(self, n_climbers: int, n_boulders: int):
        super().__init__()

        self.climber_ability = nn.Embedding(n_climbers, 1)
        self.climber_prolificity = nn.Embedding(n_climbers, 1)
        self.boulder_difficulty = nn.Embedding(n_boulders, 1)
        self.boulder_popularity = nn.Embedding(n_boulders, 1)

        self.beta = nn.Parameter(torch.tensor(1.0))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        self.mu = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.climber_ability.weight, std=0.1)
        nn.init.normal_(self.climber_prolificity.weight, std=0.1)
        nn.init.normal_(self.boulder_difficulty.weight, std=0.1)
        nn.init.normal_(self.boulder_popularity.weight, std=0.1)

    def forward(self, climber_idx: torch.Tensor, boulder_idx: torch.Tensor):
        theta = self.climber_ability(climber_idx).squeeze(-1)
        alpha = self.climber_prolificity(climber_idx).squeeze(-1)
        d = self.boulder_difficulty(boulder_idx).squeeze(-1)
        pi = self.boulder_popularity(boulder_idx).squeeze(-1)

        diff = theta - d

        gamma = torch.exp(self.log_gamma)
        logit_try = alpha + pi - gamma * (diff - self.mu)**2
        logit_send = diff
        logit_flash_given_send = diff - self.beta

        p_try = torch.sigmoid(logit_try)
        p_send_given_try = torch.sigmoid(logit_send)
        p_flash_given_send = torch.sigmoid(logit_flash_given_send)

        p_not_try = 1.0 - p_try
        p_try_fail = p_try * (1.0 - p_send_given_try)
        p_try_send = p_try * p_send_given_try * (1.0 - p_flash_given_send)
        p_try_flash = p_try * p_send_given_try * p_flash_given_send

        log_probs = torch.stack(
            [p_not_try, p_try_fail, p_try_send, p_try_flash], dim=-1
        ).clamp(min=1e-8).log()

        return log_probs
