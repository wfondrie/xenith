"""
Defines the command line functionality of Xenith
"""
import os.path
import logging
import pandas as pd

import xenith
from xenith.config import Config

def main():
    # Get command line arguments
    config = Config()

    print(config._namespace)
    print(config.verbosity)

    # Setup logging
    verbosity_dict = {0: logging.ERROR,
                      1: logging.WARNING,
                      2: logging.INFO,
                      3: logging.DEBUG}

    logging.basicConfig(format=("{asctime} [{levelname}] "
                                "{module}.{funcName} : {message}"),
                        style="{", level=verbosity_dict[config.verbosity])

    if config.command == "predict":

        logging.info("Loading model from %s.", config.model_file)
        if config.model == "custom":
            model = xenith.load_model(config.model_file)
        elif config.model == "custom_percolator":
            model = xenith.from_percolator(config.model_file)
        else:
            raise RuntimeError("Not yet implemented.")

        file_txt = ", ".join(config.psm_files)
        logging.info("Assessing PSMs from %s.", file_txt)
        prediction = model.predict(config.psm_files)

        logging.info("Estimating q-values.")
        results = prediction.estimate_qvalues("xenith_score")

        logging.info("Writing results to '%s'.", config.output_dir)
        out_files = [".psms.tsv", ".peptides.tsv", ".cross-links.tsv"]

        for result, out_file in zip(results, out_files):
            out_file = os.path.join(config.output_dir,
                                    config.fileroot + out_file)
            result.to_csv(out_file, sep="\t", index=False)



    elif config.command == "fit":
        logging.info("fitting")




if __name__ == "__main__":
    main()
