import pytest
from src.algo_orla.cispo import CISPO
from src.algo_orla.dapo import DAPO
from src.algo_orla.dpo import DPO
from src.algo_orla.icepop import IcePop
import torch


from algo_orla import PPO, Reinforce, GRPO


def test_reinforce():
    algo = Reinforce()
    got = algo.advantages(
        torch.tensor([-2.0, 1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 1.0, 1.0])
    )
    assert torch.allclose(got, torch.tensor([-3.0, 0.0, -1.0, 0.0]))


def test_ppo_ratio():
    algo = PPO()
    pi_curent = torch.tensor([0.3, 0.2, 0.2, 0.3])
    pi_old = torch.tensor([0.3, 0.1, 0.3, 0.2])
    got = algo.ratio(torch.log(pi_curent), torch.log(pi_old))
    assert torch.allclose(got, torch.tensor([1.0, 2.0, 2 / 3, 1.5]))


def test_ppo_clip():
    algo = PPO()
    ratio = torch.tensor([1.0, 2.0, 2 / 3, 1.5])
    advantage = torch.tensor(2)
    got = algo.clipped_objective(ratio, advantage, 0.2)
    assert torch.allclose(got, torch.tensor([2.0, 2.4, 4 / 3, 2.4]))


def test_grpo():
    algo = GRPO()
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    got = algo.group_advantages(rewards)
    assert torch.allclose(got, torch.tensor([[1.0, -1.0, 1.0, -1.0]]), atol=1e-6)


def test_dapo_mask():
    algo = DAPO()
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    got = algo.effective_group_mask(rewards)
    assert torch.equal(got, torch.tensor([True]))


def test_dapo_token_mean():
    algo = DAPO()
    token_loses = torch.tensor([[[-1.1, 1.0, 0.5, 0.0]]])
    mask = torch.tensor([[[True, False, True, True]]])
    got = algo.token_mean(token_loses, mask)
    assert torch.allclose(got, torch.tensor(-0.2))


def test_cispo_detached_weight():
    algo = CISPO()
    pi_curent = torch.tensor([0.3, 0.2, 0.2, 0.3])
    pi_old = torch.tensor([0.3, 0.1, 0.3, 0.2])
    low = 0.4
    high = 0.28

    got = algo.detached_weight(torch.log(pi_curent), torch.log(pi_old), low, high)
    assert torch.allclose(got, torch.tensor([1, 1.28, 2 / 3, 1.28]))


def test_icepop_mask():
    algo = IcePop()

    rollout = torch.log(torch.tensor([0.10, 0.02, 0.40, 0.005]))
    train_old = torch.log(torch.tensor([0.12, 0.20, 0.10, 0.001]))
    response_mask = torch.ones(4)
    got = algo.discrepancy_mask(train_old, rollout, response_mask, 0.5, 5.0)
    assert torch.equal(got, torch.tensor([True, False, False, False]))


def test_dpo_logit():
    algo = DPO()

    pi_ch = torch.tensor([-8.0])
    pi_rej = torch.tensor([-10.0])
    ref_ch = torch.tensor([-7.5])
    ref_rej = torch.tensor([-9.8])
    got = algo.preference_logit(pi_ch, pi_rej, ref_ch, ref_rej, 0.2)
    assert torch.allclose(got, torch.tensor([-0.06]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
