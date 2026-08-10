from __future__ import annotations

import torch


class DPO:
    def preference_logit(
        self,
        policy_chosen: torch.Tensor,  # [B, G, T]
        policy_rejected: torch.Tensor,  # [B, G, T]
        ref_chosen: torch.Tensor,  # [B, G, T]
        ref_rejected: torch.Tensor,  # [B, G, T]
        beta: float,
    ) -> torch.Tensor:  # [B, G, T]
        """beta * [(policy_chosen-ref_chosen) - (policy_rejected-ref_rejected)]

        Args:
            policy_chosen (torch.Tensor): _description_
            policy_rejected (torch.Tensor): _description_
            ref_chosen (torch.Tensor): _description_
            ref_rejected (torch.Tensor): _description_
            beta (float): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: _description_
        """
        reference_relative_margin = (policy_chosen - ref_chosen) - (
            policy_rejected - ref_rejected
        )
        return reference_relative_margin * beta
