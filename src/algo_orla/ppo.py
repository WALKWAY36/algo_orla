from __future__ import annotations

import torch


class PPO:
    def ratio(
        self,
        new_log_probs: torch.Tensor,  # [B, G, T]
        old_log_probs: torch.Tensor,  # [B, G, T]
    ) -> torch.Tensor:  # [B,G,T]
        result_ratio = torch.exp(new_log_probs - old_log_probs)  # [B,G,T]
        old_log_probs.detach()  # Чтобы gradient не шёл в old policy
        return result_ratio

    def clipped_objective(
        self,
        ratio: torch.Tensor,  # [B, G, T]
        advantage: torch.Tensor,  # [B, G]
        epsilon: float,
    ) -> torch.Tensor:
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
        surrogate = torch.min(unclipped, clipped)
        return surrogate
