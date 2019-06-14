"""
Defines the command line functionality of Xenith
"""
import logging

from xenith.config import Config

def main():
    # Get command line arguments
    config = Config()

    print(config._namespace)

    # Setup logging
    verbosity_dict = {0: logging.ERROR,
                      1: logging.WARNING,
                      2: logging.INFO,
                      3: logging.DEBUG}

    logging.basicConfig(format=("{asctime} [{levelname}] "
                                "{module}.{funcName} : {message}"),
                        style="{", level=verbosity_dict[config.verbosity])

if __name__ == "__main__":
    main()
