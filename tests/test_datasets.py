"""
Test the functionality of the 'PsmDataset' class and the auxiliary
functions in the 'dataset' module
"""
import os
import io
import pytest
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
    xenith.dataset._parse_psms(psm_txt[0], ["scannr"])

    # Try reading multiple files
    xenith.dataset._parse_psms(psm_txt[0:2], ["scannr"])

    # Try reading multiple, but columns don't match
    with pytest.raises(RuntimeError):
        xenith.dataset._parse_psms(psm_txt, ["scannr"])

    # Try reading a file, but it doesn't have a required column
    with pytest.raises(RuntimeError):
        xenith.dataset._parse_psms(psm_txt[0], ["blah"])
