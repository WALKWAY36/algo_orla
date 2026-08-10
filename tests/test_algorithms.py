import pytest
import torch


from algo_orla import PPO, Reinforce


def test_practice_reinforce():
    algo = PPO()
    got = algo.ReinforcePractice().advantages(torch.tensor([4.0]), 5.0)
    assert torch.allclose(got, torch.tensor([-1.0]))


def test_practice_ppo_ratio():
    algo = PPO()
    got = algo.ratio(torch.log(torch.tensor([0.3])), torch.log(torch.tensor([0.2])))
    assert torch.allclose(got, torch.tensor([1.5]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
