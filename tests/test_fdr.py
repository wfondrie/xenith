"""
Test that the 'fdr' module is correct.

This file does **not** test FDR functionality of the 'datasets' module.
"""
import pytest
import numpy as np
import pandas as pd

import xenith.fdr

@pytest.fixture
def dataset():
    """
    Construct a contrived dataset for testing the FDR

    By design, the dataset has several edge cases:
    1) The top score is a decoy (should result in div by 0)
    2) At one point decoy-decoy hits outnumber target-decoy hits. Under
       the Walzthoeni FDR calculation, this would be negative.
    3) There are meaningful ties

    Additionally, the order is randomized. I'm not going to set the seed
    because it should not matter what order they are in.
    """
    num_targets = [1, 1, 0, 1, 2, 2, 1, 1, 1, 2, 0, 0, 2, 1]
    score = [1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11]
    qvalues = [1, 0.75] + [0.5] * 7 + [0] * 5

    dat = pd.DataFrame({"num_targets": num_targets,
                        "score": score,
                        "qvalues": qvalues})
    dat = dat.sample(frac=1).reset_index()

    return dat


def test_descending_score(dataset):
    """
    Test that q-values for descending scores (higher == better) are
    correct.
    """
    qvals = xenith.fdr.qvalues(num_targets=dataset.num_targets.values,
                               metric=dataset.score.values,
                               desc=True)

    np.testing.assert_array_equal(dataset.qvalues.values, qvals)

    bad_qvals = xenith.fdr.qvalues(num_targets=dataset.num_targets.values,
                                   metric=dataset.score.values,
                                   desc=False)

    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_equal,
                             dataset.qvalues.values,
                             bad_qvals)


def test_ascending_score(dataset):
    """
    Test that q-values for ascending scores (lower == better) are
    correct.
    """
    neg_scores = -1 * dataset.score.values
    qvals = xenith.fdr.qvalues(num_targets=dataset.num_targets.values,
                               metric=neg_scores,
                               desc=False)

    np.testing.assert_array_equal(dataset.qvalues.values, qvals)


    bad_qvals = xenith.fdr.qvalues(num_targets=dataset.num_targets.values,
                                   metric=neg_scores,
                                   desc=True)

    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_equal,
                             dataset.qvalues.values,
                             bad_qvals)


def test_errors(dataset):
    """
    Test that errors arise in the following scenarios:
    1) 'metric' and 'num_targets' are different sizes
    """
    targ = np.random.randint(0, 2, size=10)
    score = np.random.normal(size=11)

    with pytest.raises(ValueError):
        xenith.fdr.qvalues(num_targets=targ, metric=score)
