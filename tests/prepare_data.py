"""
This script prepares the data for testing.
"""
import subprocess
import numpy as np
import pandas as pd
import xenith
import xenith.convert

np.random.seed(42)

PERCOLATE = True

def main(percolate: bool):
    """
    Creates a xenith tsv and a Percolator PIN file.

    The data used for this is a very small set of PSMs, so that it is
    readily stored on GitHub. To generate a PIN file that would
    successfully complete Percolator, I copied the same PSMs 10 times,
    and changed the 'Score' column to be drawn from two normal
    distributions.

    The Percolator command to use this should be:
    $ percolator --weights=data/weights.txt -Y --override \
      --default direction=Score data/test_large.pin


    Successful completion of this function will result in the following
    new files in the `./data` subdirectory:
    "test.tsv"
        Input for xenith in the tab-delimited format.
    "test.pin"
        A small pin file for Percolator.
    "test_large.pin"
        A large pin file that can actually be used with Percolator
    "weights.txt"
        The weights output from Percolator. Note this is present only
        if Percolator is run.

    Parameters
    ----------
    percolate : bool
        If `True`, Percolator will be run using `subprocess.run()`.
        Note that Percolator must be installed and in your path to
        execute successfully.
    """
    tsv = xenith.convert_kojak(kojak="data/test.kojak.txt",
                               perc_inter="data/test.perc.inter.txt",
                               perc_intra="data/test.perc.intra.txt",
                               out_file="data/test.tsv",
                               to_pin=False)

    pin = xenith.convert_kojak(kojak="data/test.kojak.txt",
                               perc_inter="data/test.perc.inter.txt",
                               perc_intra="data/test.perc.intra.txt",
                               out_file="data/test.pin",
                               to_pin=True)

    # Need a larger pin file for Percolator:
    pin_df = xenith.convert._read_percolator(pin)
    pin_df = pd.concat([pin_df] * 10, sort=False)

    targets = pin_df.Label == 1
    pos_scores = np.random.normal(0.5, targets.sum())
    neg_scores = np.random.normal(0, ((targets-1)**2).sum())
    pin_df.Score = pin_df.Score.replace(targets, pos_scores)
    pin_df.Score = pin_df.Score.replace(~targets, neg_scores)

    pin_file = "data/test_large.pin"
    xenith.convert._write_pin(pin_df, pin_file)

    # Percolator command
    cmd = ["percolator", "--weights=data/weights.txt", "-Y", "--override",
           "--default-direction=Score", pin_file]

    if percolate:
        proc = subprocess.run(cmd)
        print(proc)


if __name__ == "__main__":
    main(PERCOLATE)
