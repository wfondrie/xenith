"""
Test that the basic model creation classes in the 'torchmods' module
function as expected.

This does not test the 'models', which depends on 'torchmods'.
"""
import pytest
import torch

import xenith.torchmods as mods

def test_mlp_creation():
    """Verifies that models are created without errors"""
    mods.MLP(input_dim=5, layers=[3, 2, 1])
    mods.MLP(input_dim=5, layers=(3, 2, 1))
    mods.MLP(input_dim=5, layers=3)

    # invalid mods
    with pytest.raises(ValueError):
        mods.MLP(0, 1)

    with pytest.raises(ValueError):
        mods.MLP(5, 0)

    with pytest.raises(ValueError):
        mods.MLP(5, [3, 2, 0])


