"""
This module tests the command line interface for xenith.
"""
import os
import subprocess

import pytest
import torch
import numpy as np
import pandas as pd

import xenith
from test_convert import kojak_files

@pytest.fixture
def mods(tmpdir):
    """Get models and their predictions."""
    perc_mod = os.path.join("tests", "data", "weights.txt")
    model = xenith.from_percolator(perc_mod)
    xenith_mod = os.path.join(tmpdir, "model.pt")
    model.save(xenith_mod)

    # Standard prediction
    np.random.seed(1)
    data_path = os.path.join("tests", "data", "test.tsv")
    dataset = xenith.load_psms(data_path)
    dataset = model.predict(dataset)
    qvals = dataset.estimate_qvalues()

    files = [os.path.join(tmpdir, "psms.txt"),
             os.path.join(tmpdir, "xlinks.txt")]

    _ = [df.to_csv(f, sep="\t", index=False) for df, f in zip(qvals, files)]
    psms, xlinks = [pd.read_csv(f, sep="\t") for f in files]

    ret_dict = {"perc": perc_mod,
                "xenith": xenith_mod,
                "data_path": data_path,
                "psms": psms,
                "xlinks": xlinks}

    return ret_dict

def test_help():
    """
    Test that getting help does not error.

    The content of the help message is not checked, to make updating
    it a little easier. This may change in the future.
    """
    subprocess.run(["xenith"], check=True)
    subprocess.run(["xenith", "-h"], check=True)
    subprocess.run(["xenith", "predict", "-h"], check=True)
    subprocess.run(["xenith", "kojak", "-h"], check=True)


def test_predict(tmpdir, mods):
    """
    Test that the predict command is working.

    Run the predict command and compare to using the python interface.
    """
    fileroot = "test"
    out_files = [fileroot + x for x in [".psms.txt", ".xlinks.txt"]]
    out_files = [os.path.join(tmpdir, f) for f in out_files]

    base_cmd = ["xenith", "predict", mods["data_path"],
                "-r", fileroot, "-o", tmpdir,  "-m"]

    subprocess.run(base_cmd + [mods["perc"]], check=True)
    subprocess.run(base_cmd + [mods["xenith"]], check=True)

    # Verify
    psms, xlinks = [pd.read_csv(f, sep="\t") for f in out_files]

    pd.testing.assert_frame_equal(psms, mods["psms"])
    pd.testing.assert_frame_equal(xlinks, mods["xlinks"])


def test_convert_kojak(tmpdir, kojak_files):
    """
    Test that the conversion of Kojak results is working.
    """
    # Python interface
    out_file = os.path.join(tmpdir, "py.txt")
    xenith.convert.kojak(kojak_files[0], kojak_files[1], kojak_files[2],
                         out_file=out_file)
    py_res = xenith.load_psms(out_file)

    # CLI
    out_file = os.path.join(tmpdir, "cli.txt")
    subprocess.run(["xenith", "kojak", "-o", out_file] + list(kojak_files),
                   check=True)
    cli_res = xenith.load_psms(out_file)

    pd.testing.assert_frame_equal(py_res.metadata, cli_res.metadata)
    pd.testing.assert_frame_equal(py_res.features, cli_res.features)

