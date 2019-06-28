"""
Test that the basic model and loss function classes in the 'torchmods'
module behave as expected.

This does not test the 'models', which depends on 'torchmods'.
"""
import pytest
import torch

import xenith.torchmods as mods

def test_mlp_creation():
    """Verifies that MLP models are created without errors"""
    mods.MLP(input_dim=5, hidden_dims=[3, 2, 1])
    mods.MLP(input_dim=5, hidden_dims=(3, 2, 1))
    mods.MLP(input_dim=5, hidden_dims=3)

    # invalid models
    with pytest.raises(ValueError):
        mods.MLP(0, 1)

    with pytest.raises(ValueError):
        mods.MLP(5, 0)

    with pytest.raises(ValueError):
        mods.MLP(5, [3, 2, 0])

    with pytest.raises(ValueError):
        mods.MLP("a", [1, 2, 3])

    with pytest.raises(ValueError):
        mods.MLP(5, [1, 2, "a"])


def test_lm_creation():
    """Verifies that linear models are created without errors"""
    mods.Linear(input_dim=5)

    # invalid models
    with pytest.raises(ValueError):
        mods.Linear(input_dim=0)

    with pytest.raises(ValueError):
        mods.Linear(input_dim="a")


def test_mlp_iscorrect():
    """Test that you get an MLP with the correct layers and dims"""
    mlp = mods.MLP(input_dim=5, hidden_dims=[3, 2, 1])

    # weights
    assert mlp.linear_0.weight.size() == torch.Size([3, 5])
    assert mlp.linear_1.weight.size() == torch.Size([2, 3])
    assert mlp.linear_2.weight.size() == torch.Size([1, 2])
    assert mlp.linear_3.weight.size() == torch.Size([1, 1])

    # bias
    assert mlp.linear_0.bias.size() == torch.Size([3])
    assert mlp.linear_1.bias.size() == torch.Size([2])
    assert mlp.linear_2.bias.size() == torch.Size([1])
    assert mlp.linear_3.bias.size() == torch.Size([1])

    # I'll eventually manually calculate a result to compare against
    # but for now, just knowing that it doesn't error will work.
    mlp(torch.FloatTensor([[5] * 5]))

def test_lm_iscorrect():
    """Test that you get actually linear model."""
    lm = mods.Linear(input_dim=3)

    assert lm.linear.weight.size() == torch.Size([1, 3])
    assert lm.linear.bias.size() == torch.Size([1])

    lm.linear.weight = torch.nn.Parameter(torch.FloatTensor([[1, 2, 3]]))
    lm.linear.bias = torch.nn.Parameter(torch.FloatTensor([4]))

    input = torch.FloatTensor([[5, 5, 5]])
    output = torch.FloatTensor([[34]])

    assert torch.allclose(lm(input), output)


def test_sigmoid_loss():
    """Test that the sigmoid loss function works correctly"""
    sig_loss = mods.SigmoidLoss()

    high = torch.tensor([1000])
    low = torch.tensor([-1000])
    mid = torch.tensor([0])

    target_high = sig_loss(high, torch.tensor([1]))
    target_low = sig_loss(low, torch.tensor([1]))
    target_mid = sig_loss(mid, torch.tensor([1]))

    decoy_high = sig_loss(high, torch.tensor([0]))
    decoy_low = sig_loss(low, torch.tensor([0]))
    decoy_mid = sig_loss(mid, torch.tensor([0]))

    correct = torch.FloatTensor([0])
    incorrect = torch.FloatTensor([1])
    midway = torch.FloatTensor([0.5])

    assert torch.allclose(target_high, correct, atol=2e-7)
    assert torch.allclose(target_low, incorrect, atol=2e-7)
    assert torch.allclose(target_mid, midway)
    assert torch.allclose(decoy_high, incorrect, atol=2e-7)
    assert torch.allclose(decoy_low, correct, atol=2e-7)
    assert torch.allclose(decoy_mid, midway)

def test_hybrid_loss():
    hybrid_loss = mods.HybridLoss()

    high = torch.tensor([1000])
    low = torch.tensor([-1000])
    mid = torch.tensor([0])

    target_high = hybrid_loss(high, torch.tensor([1]))
    target_low = hybrid_loss(low, torch.tensor([1]))
    target_mid = hybrid_loss(mid, torch.tensor([1]))

    decoy_high = hybrid_loss(high, torch.tensor([0]))
    decoy_low = hybrid_loss(low, torch.tensor([0]))
    decoy_mid = hybrid_loss(mid, torch.tensor([0]))

    correct = torch.FloatTensor([0])
    incorrect_target = torch.FloatTensor([1])
    midway_target = torch.FloatTensor([0.5])
    incorrect_decoy = torch.FloatTensor([15.9424])
    midway_decoy = torch.FloatTensor([0.6931])

    assert torch.allclose(target_high, correct, atol=2e-7)
    assert torch.allclose(target_low, incorrect_target, atol=2e-7)
    assert torch.allclose(target_mid, midway_target)
    assert torch.allclose(decoy_high, incorrect_decoy, atol=1e-4)
    assert torch.allclose(decoy_low, correct, atol=2e-7)
    assert torch.allclose(decoy_mid, midway_decoy, atol=1e-4)
