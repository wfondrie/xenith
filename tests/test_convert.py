"""
Test the converters.

Currently, there is only the Kojak converter.
"""
import os

import pytest
import numpy as np
import pandas as pd

import xenith
from xenith.convert.kojak import _count_proteins
from xenith.convert.kojak import _all_decoy
from xenith.convert.kojak import _read_percolator
from xenith.convert.kojak import _read_kojak
from xenith.convert.kojak import _write_pin

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


def test_count_proteins():
    """Test that count proteins works as expected."""
    proteins = pd.DataFrame({"ProteinA": ["a", "b", "c", "a;b"],
                             "ProteinB": ["c", "a;c", "b", "b;c;a"]})

    num_prot = _count_proteins(proteins)
    assert num_prot == 3

def test_all_decoy():
    """Test that the _all_decoy function works."""
    proteins = pd.Series(["a_b", "d_c", "a_b;d_c", "d_a;d_b", "d_c;a_a"])
    answer = pd.Series([0, 1, 0, 1, 0])

    assert answer.equals(_all_decoy(proteins, "d_"))

def test_read_percolator(tmpdir):
    """Verify reading PIN files works as expected."""
    pin_file = os.path.join(tmpdir, "test.pin")
    with open(pin_file, "w") as pin:
        pin.write("col1\tcol2\na\tb\tc\td")

    pin_df = _read_percolator(pin_file)
    answer = pd.DataFrame({"col1": ["a"],
                           "col2": ["b\tc\td"]})

    assert answer.equals(pin_df)

def test_read_kojak():
    """Verify that reading kojak.txt files works as expected."""
    kojak_file = os.path.join("tests", "data", "test.kojak.txt")
    kojak_df = _read_kojak(kojak_file, "decoy_")

    psm = kojak_df[0].loc[1657, :]
    assert psm.Peptide == "-.MEPKF(4)--RNKKF(4).-"
    assert psm.ProteinB == ("wef|PV4545|PPARg-LBD_human GST-tagged PPARgamma"
                            " LBD;decoy_wef|PV4545|PPARg-LBD_human")
    assert psm.NumTarget == 2

    print(kojak_df[1])
    assert set(kojak_df[1]) == set(["scannr", "Peptide", "Label"])


def test_write_pin(tmpdir):
    """Test the _write_pin() utility function."""
    base_pin = os.path.join("tests", "data", "test.pin")
    base_df = _read_percolator(base_pin)

    new_pin = os.path.join(tmpdir, "new.pin")
    _write_pin(base_df, new_pin)

    new_df = _read_percolator(new_pin)

    pd.testing.assert_frame_equal(new_df, base_df)
