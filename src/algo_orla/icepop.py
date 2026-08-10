from __future__ import annotations

import torch


class IcePop:
    def discrepancy_mask(
        self,
        train_old_log_probs: torch.Tensor,  # [B, G, T]
        rollout_log_probs: torch.Tensor,  # [B, G, T]
        response_mask: torch.Tensor,  # [B, G, T]
        ratio_min: float,
        ratio_max: float,
    ) -> torch.Tensor:  # [B, G, T]
        """exp(train_old_logp - rollout_logp), then double-sided mask.

        Args:
            train_old_log_probs (torch.Tensor): _description_
            rollout_log_probs (torch.Tensor): _description_
            response_mask (torch.Tensor): _description_
            ratio_min (float): _description_
            ratio_max (float): _description_

        Returns:
            torch.Tensor: _description_
        """
        discrepancy_ratio = train_old_log_probs - rollout_log_probs  # [B, G, T]
        discrepancy = torch.exp(discrepancy_ratio)  # [B, G, T]
        result_mask = (
            (discrepancy <= ratio_max)
            & (discrepancy >= ratio_min)
            & response_mask.bool()
        )  # [B, G, T]

        return result_mask
