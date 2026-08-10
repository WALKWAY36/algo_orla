from __future__ import annotations

import torch


class Reinforce:
    def advantages(
        self,
        returns: torch.Tensor,  # [B, G]
        baseline: torch.Tensor | float,  # [B, G]
    ) -> torch.Tensor:  # [B, G]
        adv = returns - baseline  # [B, G]
        adv.detach()  # Чтобы policy loss не обучал critic
        return adv
