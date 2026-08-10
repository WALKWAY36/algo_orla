from __future__ import annotations

import torch


class GRPO:
    def group_advantages(
        self,
        rewards: torch.Tensor,  # [B, G]
        eps: float = 1e-8,
    ) -> torch.Tensor:  # [B, G]
        """normalize each row by group mean and population std.

        Args:
            rewards (torch.Tensor): _description_
            eps (float, optional): _description_. Defaults to 1e-8.

        Returns:
            torch.Tensor: _description_
        """
        baseline = rewards.mean(dim=1, keepdim=True)  # [B, G]
        std = rewards.std(dim=1, keepdim=True, correction=0)  # [B, G]

        return (rewards - baseline) / (std - eps).abs()
