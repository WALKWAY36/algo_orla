from __future__ import annotations

import torch


class DAPO:
    def effective_group_mask(
        self,
        rewards: torch.Tensor,  # [B]
        eps: float = 1e-8,
    ) -> torch.Tensor:  # [B]
        """keep only groups with useful reward variance.

        Args:
            rewards (torch.Tensor): _description_
            eps (float, optional): _description_. Defaults to 1e-8.

        Returns:
            torch.Tensor: _description_
        """

        std = rewards.std(dim=1, correction=0)  # [B]

        return std > eps

    def token_mean(
        self,
        token_losses: torch.Tensor,  # [B, G, T]
        mask: torch.Tensor,  # [B, G, T]
    ) -> torch.Tensor:  # []
        """unlike sequence mean, every valid token has equal weight.

        Args:
            token_losses (torch.Tensor): _description_
            mask (torch.Tensor): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """
        mask = mask.to(token_losses.float().dtype)  # [B, G, T]
        nominator = token_losses * mask  # [B, G, T]
        return nominator.sum(dim=-1) / mask.sum(dim=-1)
