"""
Test the converters.

Currently, there is only the Kojak converter.
"""
import os

import pytest
import numpy as np
import pandas as pd

import xenith

@pytest.fixture
def kojak_files():
    """Locations of test Kojak files"""
    kojak = os.path.join("tests", "data", "test.kojak.txt")
    intra = os.path.join("tests", "data", "test.perc.intra.txt")
    inter = os.path.join("tests", "data", "test.perc.inter.txt")

    return (kojak, inter, intra)

def test_convert_kojak(tmpdir, kojak_files):
    """
    Test the conversion of Kojak results to xenith tab-delimited format.
    """
    out_file = os.path.join(tmpdir, "test.txt")
    xenith.convert.kojak(kojak_files[0], kojak_files[1], kojak_files[2],
                         out_file=out_file)

    # Because there are only 2 proteins in these results, 'intraprotein'
    # should always be 0.
    dataset = xenith.load_psms(out_file)
    intraprotein = dataset.features.intraprotein.tolist()

    assert all(not x for x in intraprotein)

