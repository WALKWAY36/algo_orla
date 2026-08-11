from __future__ import annotations

import torch


class PPO:
    def ratio(
        self,
        new_log_probs: torch.Tensor,  # [B, G, T]
        old_log_probs: torch.Tensor,  # [B, G, T]
    ) -> torch.Tensor:  # [B,G,T]
        old = old_log_probs.detach()  # Чтобы gradient не шёл в old policy
        result_ratio = torch.exp(new_log_probs - old)  # [B,G,T]
        return result_ratio

    def clipped_objective(
        self,
        ratio: torch.Tensor,  # [B, G, T]
        advantage: torch.Tensor,  # [B, G]
        epsilon: float,
    ) -> torch.Tensor:
        adv = advantage.unsqueeze(-1)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
        surrogate = torch.min(unclipped, clipped)
        return surrogate
