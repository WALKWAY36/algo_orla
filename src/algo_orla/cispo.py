from __future__ import annotations

import torch


class CISPO:
    def detached_weight(
        self,
        new_log_probs: torch.Tensor,  # [B, G, T]
        old_log_probs: torch.Tensor,  # [B, G, T]
        low: float,
        high: float,
    ) -> torch.Tensor:  # [B, G, T]
        """ratio -> clip -> detach.

        Args:
            new_log_probs (torch.Tensor): _description_
            old_log_probs (torch.Tensor): _description_
            low (float): _description_
            high (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """

        ratio = torch.exp(new_log_probs - old_log_probs)  # [B, G, T]
        weight = torch.clamp(ratio, 1 - low, 1 + high)  # [B, G, T]

        return weight.detach()
