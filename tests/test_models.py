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
    assert np.array_equal(loaded.hidden_dims, orig.hidden_dims)

    for params in zip(loaded.model.parameters(), orig.model.parameters()):
        assert torch.allclose(params[0], params[1])


def test_from_percolator():
    """
    Test the from_percolator() function.

    Verify that a model is correctly loaded from the Percolator weights
    output.
    """
    path = os.path.join("tests", "data", "weights.txt")
    loaded = xenith.from_percolator(path)

    # Correct answers for feat_mean and feat_stdev
    features = ['Score', 'eVal', 'eValA', 'eValB', 'IonMatch', 'ConIonMatch',
                'IonMatchA', 'ConIonMatchA', 'IonMatchB', 'ConIonMatchB',
                'PPScoreDiff', 'Mass', 'PPM', 'LenSum', 'intraprotein',
                'LenRat', 'Charge_1', 'Charge_2', 'Charge_3', 'Charge_4',
                'Charge_5', 'Charge_6', 'Charge_7', 'Charge_8']
    correct_feat = pd.Series([0]*24, index=features)

    assert loaded.source == "percolator"
    assert loaded.num_features == 24
    assert loaded.pretrained == True
    assert loaded.feat_mean.equals(correct_feat)
    assert loaded.feat_stdev.equals(correct_feat)


def test_count_parameters():
    """
    Test the XenithModel.count_parameters() method.

    Verify that the correct number of parameters is returned for a linear
    and MLP model.
    """
    linear_model = xenith.new_model(3, None)
    mlp_model = xenith.new_model(3, [2])

    linear_params = linear_model.count_parameters()
    mlp_params = mlp_model.count_parameters()

    print(linear_params)
    print(mlp_params)

    assert linear_params == 4 # 4 feat + 1 bias
    assert mlp_params == 11 # (4x2 weights + 2 bias) + 1 bias

def test_fit_rough():
    """
    Test that the XenithModel.fit() method does not error.

    Verify that the fit method works, but don't check for correctness.
    These tests will also take the longest.
    """
    dataset = xenith.load_psms(os.path.join("tests", "data", "test.tsv"))

    linear_model = xenith.new_model(dataset.num_features(), None)
    mlp_model = xenith.new_model(dataset.num_features(), [2])

    _ = linear_model.fit(dataset, dataset, batch_size=1, max_epochs=10)
    _ = mlp_model.fit(dataset, dataset, batch_size=1, max_epochs=10)


def test_predict():
    """
    Test the XenithModel.predict() method.
    """
    pass


