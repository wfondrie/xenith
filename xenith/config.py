"""
Contains all the the configuration details for running xenith from the
command-line.
"""
import argparse
import textwrap
#import configargparse


class XenithHelpFormatter(argparse.HelpFormatter):
    """Format help text to keep newlines and whitespace"""
    def _fill_text(self, text, width, indent):
        text_list = text.splitlines(keepends=True)
        return "\n".join(_process_line(l, width, indent) for l in text_list)

class Config():
    """
    The xenith configuration options.

    Options can be specified as command-line arguments.
    """
    def __init__(self) -> None:
        """Initialize configuration values."""
        desc = ("xenith: Enhanced cross-linked peptide detection using "
                "pretrained models.\n"
                "============================================================"
                "============\n \n"
                "Official code website: "
                "https://github.com/wfondrie/xenith")

        docs = ("For more detailed documentation, examples, and explanations, "
                "see the official documentation website: "
                "https://xenith.readthedocs.com")

        # Setup main parser
        parser = argparse.ArgumentParser(description=desc + "\n \n",
                                         epilog=docs,
                                         formatter_class=XenithHelpFormatter)

        # Setup parser for shared arguments
        common = argparse.ArgumentParser(add_help=False)

        common.add_argument("--verbosity",
                            default=2,
                            type=int,
                            choices=[0, 1, 2],
                            help=("Specify the verbosity of the current "
                                  "process. Each level prints the following "
                                  "messages, including all those at a lower "
                                  "verbosity: 0-errors, 1-warnings, 2-messages"
                                  ", 3-debug info."))

        # Setup subparsers for each subcommand
        subparser = parser.add_subparsers(title="Command", dest="command")

        # Evaluation
        eval_help = ("Use a previously trained model to assess a new "
                     "collection of peptide-spectrum matches (PSMs).")

        eval_parser = subparser.add_parser("assess",
                                           parents=[common],
                                           description=eval_help,
                                           epilog=docs,
                                           help=eval_help)

        eval_parser.add_argument("psm_file",
                                 type=str,
                                 nargs=1,
                                 help=("A collection of cross-linked PSMs in"
                                       "the xenith tab-separated format."))

        eval_parser.add_argument("--output_dir",
                                 default=".",
                                 type=str,
                                 help=("The directory in which to write xenith"
                                       " results. This is the working "
                                       "directory by default."))

        eval_parser.add_argument("--fileroot",
                                 type=str,
                                 help=("The fileroot string will be added as a"
                                       " prefix to all output file names. This"
                                       " is the 'psm_file' root by default."))

        eval_parser.add_argument("--model",
                                 type=str,
                                 required=True,
                                 help=("Which model should be used? The built"
                                       "-in models are 'kojak_mlp' and "
                                       "'kojak_percolator' which are both for "
                                       "output from Kojak. Custom models can "
                                       "be created using the 'train' "
                                       "command or using a Percolator "
                                       "weights file. If a custom model is "
                                       "used this should be the path to the "
                                       "model file."))

        # Train
        train_help = ("Train a new model to assess future collections of "
                      "cross-linked peptide-spectrum matches (PSMs).")
        train_parser = subparser.add_parser("train",
                                            parents=[common],
                                            description=train_help,
                                            epilog=docs,
                                            help=train_help)

        train_parser.add_argument("psm_files",
                                  type=str,
                                  nargs="+",
                                  help=("One or more collections of PSMs in "
                                        "the xenith tab-separated format"))

        train_parser.add_argument("--hidden_dims",
                                  type=str,
                                  nargs=1,
                                  help=("A comma-separated list indicating "
                                        "dimensions of hidden layers to use "
                                        "in the model. This is '8,8,8' by "
                                        "default, which defines three hidden "
                                        "layers with 8 neurons each"))

        train_parser.add_argument("--max_epochs",
                                  type=int,
                                  default=200,
                                  nargs=1,
                                  help=("The maximum number of epochs "
                                        "used for model training."))

        train_parser.add_argument("--learning_rate",
                                  type=float,
                                  default=0.0001,
                                  nargs=1,
                                  help=("The learning rate to be used for "
                                        "optimization with Adam. The default "
                                        "is 0.0001"))

        train_parser.add_argument("--batch_size",
                                  type=int,
                                  default=128,
                                  nargs=1,
                                  help=("The batch size to use for "
                                        "optimization with Adam. The default "
                                        "is 128."))

        train_parser.add_argument("--fileroot",
                                  type=str,
                                  default="xenith_model",
                                  nargs=1,
                                  help=("The fileroot string will be added as "
                                        "a prefix to all output file names. "
                                        "This is 'xenith_model' by default."))

        self._namespace = vars(parser.parse_args())

    def __getattr__(self, option):
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)

# Utility Functions -----------------------------------------------------------
def _process_line(line, width, indent):
    line = textwrap.fill(line, width, initial_indent=indent,
                         subsequent_indent=indent,
                         replace_whitespace=False)
    return line.strip()
