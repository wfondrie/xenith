"""
Defines the command line functionality of Xenith
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

import xenith
from xenith.config import Config

def main():
    """The command line entry point."""
    # Get command line arguments
    config = Config()

    # If no args are present, show help and exit.
    if len(sys.argv) == 1:
        config.parser.print_help(sys.stderr)
        quit()

    # Setup logging
    verbosity_dict = {0: logging.ERROR,
                      1: logging.WARNING,
                      2: logging.INFO,
                      3: logging.DEBUG}

    logging.basicConfig(format=("{asctime} [{levelname}] "
                                "{module}.{funcName} : {message}"),
                        style="{", level=verbosity_dict[config.verbosity])

    if config.command == "predict":
        np.random.seed(config.seed)
        dataset = xenith.load_psms(config.psm_files)

        try:
            model = xenith.from_percolator(config.model)
        except UnicodeDecodeError:
            model = xenith.load_model(config.model)

        pred = model.predict(dataset)
        dataset.add_metric(pred, name="score")
        psms, xlinks = dataset.estimate_qvalues("score")

        out_base = os.path.join(config.output_dir, config.fileroot)
        psms.to_csv(out_base + ".psms.txt", sep="\t", index=False)
        xlinks.to_csv(out_base + ".xlinks.txt", sep="\t", index=False)

    elif config.command == "kojak":
        xenith.convert.kojak(kojak_txt=config.kojak_txt,
                             perc_inter=config.perc_inter,
                             perc_intra=config.perc_intra,
                             version=config.version,
                             out_file=config.output_file,
                             max_charge=config.max_charge,
                             to_pin=config.to_pin)
