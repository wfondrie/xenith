"""
Test the constructors and use of the XenithModel class.

These are found in 'models.py'
"""
import os
import copy
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
    return os.path.join("tests", "data", "weights.txt")

@pytest.fixture
def input_tsv():
    """A small collection of PSMs in the xenith TSV format."""
    return os.path.join("tests", "data", "test.tsv")

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

    # also Linear
    orig = xenith.new_model(5, [])
    orig.save(path)
    loaded = xenith.load_model(path)

    for params in zip(loaded.model.parameters(), orig.model.parameters()):
        assert torch.allclose(params[0], params[1])

    with pytest.raises(FileNotFoundError):
        xenith.load_model("blah")


def test_from_percolator(perc_weights):
    """
    Test the from_percolator() function.

    Verify that a model is correctly loaded from the Percolator weights
    output.
    """
    path = os.path.join(perc_weights)
    loaded = xenith.from_percolator(path)

    # Correct answers for feat_mean and feat_stdev
    features = ['score', 'eval', 'evala', 'evalb', 'ionmatch', 'conionmatch',
                'ionmatcha', 'conionmatcha', 'ionmatchb', 'conionmatchb',
                'ppscorediff', 'mass', 'ppm', 'lensum', 'intraprotein',
                'lenrat', 'charge_1', 'charge_2', 'charge_3', 'charge_4',
                'charge_5', 'charge_6', 'charge_7', 'charge_8']
    correct_feat = pd.Series([0]*24, index=features)

    assert loaded.source == "percolator"
    assert loaded.num_features == 24
    assert loaded.pretrained == True
    assert loaded.feat_mean.equals(correct_feat)
    assert loaded.feat_stdev.equals(correct_feat)

    with pytest.raises(FileNotFoundError):
        xenith.from_percolator("blah")


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

def test_fit_rough(input_tsv):
    """
    Test that the XenithModel.fit() method does not error.

    Verify that the fit method works, but don't check for correctness.
    These tests will also take the longest.
    """
    dataset = xenith.load_psms(input_tsv)

    linear_model = xenith.new_model(dataset.num_features(), None)
    mlp_model = xenith.new_model(dataset.num_features(), [2])

    _ = linear_model.fit(dataset, dataset, batch_size=1, max_epochs=10)
    _ = mlp_model.fit(dataset, dataset, batch_size=1, max_epochs=10)

    # force early stop
    silly_val = copy.deepcopy(dataset)
    silly_val.features["eval"] = np.random.normal(size=len(silly_val))
    silly_val.features["evala"] = np.random.normal(size=len(silly_val))
    silly_val.features["evalb"] = np.random.normal(size=len(silly_val))
    silly_val.features["score"] = np.random.normal(size=len(silly_val))
    silly_val.features["ppscorediff"] = np.random.normal(size=len(silly_val))
    silly_val.features["ionmatch"] = np.random.normal(size=len(silly_val))
    silly_val.features["intraprotein"] = 1

    loss_df = linear_model.fit(dataset, silly_val, batch_size=1, early_stop=1)
    assert len(loss_df) < 100

@pytest.fixture
def contrived_dataset(tmpdir):
    """Create a simple dataset to test model predictions"""
    dset = pd.DataFrame({"psmid": [1, 2, 3],
                         "numtarget": [0, 1, 2],
                         "scannr": [1, 2, 3],
                         "peptidea": ["a", "b", "c"],
                         "peptideb": ["d", "e", "f"],
                         "peptidelinksitea": [1, 1, 1],
                         "peptidelinksiteb": [2, 2, 2],
                         "proteinlinksitea": [1, 1, 1],
                         "proteinlinksiteb": [2, 2, 2],
                         "proteina": ["a", "b", "c"],
                         "proteinb": ["a", "b", "c"],
                         "feat_a": [0, 1, 2],
                         "feat_b": [3, 4, 5]})

    feat_mean = pd.Series([1, 4], index=["feat_a", "feat_b"])
    feat_stdev = pd.Series([np.std([0, 1, 2], ddof=0)]*2, index=["feat_a", "feat_b"])

    dset_file = os.path.join(tmpdir, "test.tsv")
    dset.to_csv(dset_file, sep="\t", index=False)

    dataset = xenith.load_psms(dset_file)

    return (dataset, feat_mean, feat_stdev)


def test_predict(contrived_dataset):
    """
    Test the XenithModel.predict() method.
    """
    dataset, feat_mean, feat_stdev = contrived_dataset

    linear_model = xenith.new_model(2, [])
    linear_model.model.linear.weight = torch.nn.Parameter(torch.FloatTensor([[1, 2]]))
    linear_model.model.linear.bias = torch.nn.Parameter(torch.FloatTensor([3]))

    # With norm.
    dataset = linear_model.predict(dataset, name="pred")
    metrics = dataset.get_metrics()

    feat_a = ((dataset.features.feat_a.values - feat_mean[0]) /
              feat_stdev[0])

    feat_b = ((dataset.features.feat_b.values - feat_mean[1]) /
              feat_stdev[1])

    expected = feat_a + 2 * feat_b + 3
    assert np.allclose(metrics.pred.values, expected)

    # Without norm.
    linear_model.source = "percolator"
    dataset = linear_model.predict(dataset, name="pred")
    metrics = dataset.get_metrics()
    expected = np.array([9, 12, 15])
    assert np.allclose(metrics.pred.values, expected)



