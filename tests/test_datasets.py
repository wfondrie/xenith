"""
Test the functionality of the 'PsmDataset' class and the auxiliary
functions in the 'dataset' module
"""
import os
import pytest
import torch
import numpy as np
import pandas as pd

import xenith.dataset

@pytest.fixture
def psm_txt(tmpdir):
    """
    Based on one file, make three varieties of xenith files.

    Elements 0 and 1 will have the same columns, but slightly
    different data.

    Element 2 will have different columns but still be valid.
    """
    np.random.seed(1)
    out_files = [os.path.join(tmpdir, str(i)) for i in range(3)]
    test_file = os.path.join("tests", "data", "test.tsv")

    base_dat = pd.read_csv(test_file, sep="\t")

    # original data
    base_dat.to_csv(out_files[0], sep="\t", index=False)

    # Modify scores a little
    base_dat.Score = base_dat.Score + np.random.normal(size=len(base_dat))
    base_dat.PsmId = base_dat.PsmId + "-mod"
    base_dat.to_csv(out_files[1], sep="\t", index=False)

    # Delete a column
    base_dat = base_dat.drop(columns="eVal")
    base_dat.to_csv(out_files[2], sep="\t", index=False)

    return out_files


def test_parsing(psm_txt):
    """Test that parsing works as expected"""
    # Try reading file
    xenith.dataset._parse_psms(psm_txt[0], ["scannr"], None)

    # Try reading multiple files
    xenith.dataset._parse_psms(psm_txt[0:2], ["scannr"], None)

    # Try reading multiple, but columns don't match
    with pytest.raises(RuntimeError):
        xenith.dataset._parse_psms(psm_txt, ["scannr"], None)

    # Try reading a file, but it doesn't have a required column
    with pytest.raises(RuntimeError):
        xenith.dataset._parse_psms(psm_txt[0], ["blah"], None)


@pytest.fixture
def toy_features():
    """
    Generate a sample feature dataframe with one column that isn't a
    feature.
    """
    feat = pd.DataFrame({"A": [1, 2, 3],
                         "B": [4, 5, 6],
                         "C": [7, 8, 9],
                         "D": ["a", "b", "c"]})

    return (feat, feat.loc[:, ["A", "B", "C"]])

def test_features(toy_features):
    """Verify basic feature processing and error checking works"""
    feat = xenith.dataset._process_features(toy_features[1],
                                            feat_mean=None,
                                            feat_stdev=None,
                                            normalize=True)

    val = 1.224745
    norm_feat = np.array([[-val]*3, [0]*3, [val]*3])
    fmean = np.array([2, 5, 8])
    fstdev = np.std([1,2,3], ddof=0)

    assert np.allclose(feat[0].values, norm_feat, atol=1e-6)
    assert np.allclose(fmean, feat[1])
    assert np.allclose(fstdev, feat[2])

    # Non-numeric columns should raise a ValueError
    with pytest.raises(ValueError):
        xenith.dataset._process_features(toy_features[0],
                                         feat_mean=None,
                                         feat_stdev=None,
                                         normalize=True)


def test_feature_norm_off(toy_features):
    """Test that 'normalization' of _process_features() works"""
    feat = xenith.dataset._process_features(toy_features[1],
                                            feat_mean=None,
                                            feat_stdev=None,
                                            normalize=False)

    assert np.allclose(feat[0].values, toy_features[1].values)

def test_feature_custom_norm(toy_features):
    """
    Test using a custom mean and standard deviation in
    _process_features() works.
    """
    fmean = pd.Series([1, 1, 1], index=["A", "B", "C"])
    fstdev = pd.Series([1, 1, 1], index=["A", "B", "C"])
    norm_feat = np.transpose(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    feat = xenith.dataset._process_features(toy_features[1],
                                            feat_mean=fmean,
                                            feat_stdev=fstdev,
                                            normalize=True)

    assert np.allclose(norm_feat, feat[0].values)
    assert np.allclose(fmean.values, feat[1].values)
    assert np.allclose(fstdev.values, feat[2].values)

def test_feature_mismatch(toy_features):
    """
    Test that discrepancies between the features and the normalization factors
    raise appropriate errors.
    """
    fmean = pd.Series([1, 1], index=["A", "B"])
    fstdev = pd.Series([1, 1], index=["A", "B"])

    with pytest.raises(RuntimeError):
        xenith.dataset._process_features(toy_features[1],
                                         feat_mean=fmean,
                                         feat_stdev=None,
                                         normalize=True)

    with pytest.raises(RuntimeError):
        xenith.dataset._process_features(toy_features[1],
                                         feat_mean=fmean,
                                         feat_stdev=fstdev,
                                         normalize=True)


def test_dataset_init(psm_txt):
    """Test that a XenithDataset is initialized properly"""
    dset1 = xenith.load_psms(psm_txt[0])
    dset2 = xenith.dataset.XenithDataset(psm_txt[0])

    assert dset1.metadata.equals(dset2.metadata)
    assert dset1.features.equals(dset2.features)
    assert dset1.metrics.equals(dset2.metrics)
    assert dset1.metadata.shape == (21, 12) # Size of the test dataset
    assert dset1.features.shape == (21, 24)
    assert dset1.metrics.shape == (0, 0)

    # Also test adding a metadata column
    dset1 = xenith.load_psms(psm_txt[0], additional_metadata=["eVal", "eVala"])
    dset2 = xenith.load_psms(psm_txt[0], additional_metadata="eval")

    assert dset1.metadata.shape == (21, 14)
    assert dset2.metadata.shape == (21, 13)

    with pytest.raises(RuntimeError):
        xenith.load_psms(psm_txt[0], additional_metadata="x")


def test_torch_dataset_init(psm_txt):
    """
    Test that a _PsmDataset is generated correctly from a
    XenithDataset.
    """
    dset = xenith.load_psms(psm_txt[0])
    psmdset1 = xenith.dataset._PsmDataset(dset, None, None, True)
    psmdset2 = xenith.dataset._PsmDataset(dset, None, None, False)

    assert tuple(psmdset1.features.shape) == dset.features.shape
    assert not torch.allclose(psmdset1.features, psmdset2.features)
    assert torch.all(torch.eq(psmdset1.target, psmdset2.target))

    # check target
    target = [x.startswith("T") for x in dset.metadata.psmid]
    target = torch.FloatTensor(target)
    assert torch.all(torch.eq(psmdset1.target, target))


def test_torch_dataset_methods(psm_txt):
    """
    Test that the methods needed for PyTorch work as expected for a
    _PsmDataset.
    """
    dset = xenith.load_psms(psm_txt[0])
    psmdset = xenith.dataset._PsmDataset(dset, None, None, True)

    assert len(psmdset) == 21

    item = (psmdset.target[1], psmdset.features[1, :])
    assert torch.all(item[0] == psmdset[1][0])
    assert torch.all(item[1] == psmdset[1][1])

def test_metrics(psm_txt):
    """Test that you can add and get metrics."""
    dset = xenith.load_psms(psm_txt[0:2])
    dset.add_metric("xcorr", dset.features.score)
    metrics = dset.get_metrics()

    assert len(metrics) == len(dset.metadata)
    assert np.array_equal(metrics.xcorr.values, dset.features.score.values)

def test_qvalues(psm_txt):
    """
    Test that the XenithDataset.estimate_qvalues() method is working at
    both the PSM and cross-link level.
    """
    dset = xenith.load_psms(psm_txt[0:2])
    dset.add_metric("XenithScore", np.random.normal(size=len(dset)))

    qvals = dset.estimate_qvalues()
    dset.estimate_qvalues(desc=False)
