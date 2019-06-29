"""
Test the constructors and use of the XenithModel class.

These are found in 'models.py'
"""
import os
import logging
import pytest
import torch
import numpy as np
import pandas as pd

import xenith
import xenith.models as mods

@pytest.fixture
def perc_weights():
    """A Percolator weights file"""
    return os.path.join("data", "weights.txt")

@pytest.fixture
def input_tsv():
    """A small collection of PSMs in the xenith TSV format."""
    return os.path.join("data", "test.tsv")

def test_device(caplog):
    """
    Kind of test _set_device(). Travis-CI doesn't have GPUs available
    so this will always be 'cpu'
    """
    caplog.set_level(logging.WARNING)
    assert mods._set_device(True) == torch.device("cpu")
    assert "No gpu" in caplog.text

def test_new_model():
    """
    Test the new_model() function.

    Verify that the number of features changes and that you get a
    linear model when hidden_dims is None or [].
    """
    mod = xenith.new_model(5)
    assert mod.model.linear_0.in_features == 5

    mod = xenith.new_model(1)
    assert mod.model.linear_0.in_features == 1

    mod = xenith.new_model(1, hidden_dims=[])
    assert type(mod.model).__name__ == "Linear"

    mod = xenith.new_model(1, hidden_dims=None)
    assert type(mod.model).__name__ == "Linear"
