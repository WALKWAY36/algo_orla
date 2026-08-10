import pytest
import torch


from algo_orla import PPO, Reinforce, GRPO


def test_reinforce():
    algo = Reinforce()
    got = algo.advantages(torch.tensor([4.0]), 5.0)
    assert torch.allclose(got, torch.tensor([-1.0]))


def test_ppo_ratio():
    algo = PPO()
    got = algo.ratio(torch.log(torch.tensor([0.3])), torch.log(torch.tensor([0.2])))
    assert torch.allclose(got, torch.tensor([1.5]))


def test_ppo_clip():
    algo = PPO()
    got = algo.clipped_objective(torch.tensor([1.5]), torch.tensor([2.0]), 0.2)
    assert torch.allclose(got, torch.tensor([2.4]))


def test_grpo():
    algo = GRPO()
    got = algo.group_advantages(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(got, torch.tensor([[1.0, -1.0, 1.0, -1.0]]), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
