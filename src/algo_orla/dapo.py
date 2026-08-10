from __future__ import annotations

import torch


class DAPO:
    def effective_group_mask(
        self,
        rewards: torch.Tensor,  # [B, G]
        eps: float = 1e-8,
    ) -> torch.Tensor:  # [B, G]
        """keep only groups with useful reward variance.

        Args:
            rewards (torch.Tensor): _description_
            eps (float, optional): _description_. Defaults to 1e-8.

        Returns:
            torch.Tensor: _description_
        """

        std = rewards.std(dim=1, keepdim=True, correction=0)  # [B, G]

        return torch.where(std > eps, bool(1), bool(0))

    def token_mean(
        self,
        token_losses: torch.Tensor,  # [B, G, T]
        mask: torch.Tensor,  # [B, G, T]
    ) -> torch.Tensor:  # [B, G, T]
        """unlike sequence mean, every valid token has equal weight.

        Args:
            token_losses (torch.Tensor): _description_
            mask (torch.Tensor): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """
        mask = mask.to(token_losses.dtype)  # [B, G, T]
        return sum(token_losses) / sum(mask)
