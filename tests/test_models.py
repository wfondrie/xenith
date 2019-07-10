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


def test_load_model(tmpdir):
    """
    Test the load_model() function.

    Verify that the loaded model is matches the original. This also
    tests the XenithModel.save() method.
    """
    path = os.path.join(tmpdir, "model.pt")
    orig = xenith.new_model(5)
    orig.save(path)

    loaded = xenith.load_model(path)

    assert loaded.source == orig.source
    assert loaded.num_features == orig.num_features
    assert loaded.pretrained == orig.pretrained
    assert loaded.feat_mean == orig.feat_mean # Both are None.
    assert loaded.feat_stdev == orig.feat_stdev # Both are None.
    assert np.allclose(loaded.hidden_dims, orig.hidden_dims)

    for params in zip(loaded.model.parameters(), orig.model.parameters()):
        assert torch.allclose(params[0], params[1])


#def test_from_percolator():
#    """
#    Test the from_percolator() function.
#
#    Verify that a model is correctly loaded from the Percolator weights
#    output.
#    """
#    path = os.path.join("tests", "data", "weights.txt")
#    loaded = xenith.from_percolator(path)

