"""Student exercises.

Every method below intentionally contains a small conceptual gap. Implement them
WITHOUT copying from src/llm_rl_lab/algorithms first. After you finish, compare.
"""

from __future__ import annotations

import torch


class ReinforcePractice:
    def advantages(
        self,
        returns: torch.Tensor,  # [B, G]
        baseline: torch.Tensor | float,  # [B, G]
    ) -> torch.Tensor:  # [B, G]
        adv = returns - baseline  # [B, G]
        adv.detach()  # Чтобы policy loss не обучал critic
        return adv


class PPOPractice:
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


class GRPOPractice:
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


class DAPOPractice:
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


class CISPOPractice:
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


class IcePopPractice:
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


class DPOPractice:
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
