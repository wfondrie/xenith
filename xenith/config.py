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
                            choices=[0, 1, 2, 3],
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

        eval_parser = subparser.add_parser("predict",
                                           parents=[common],
                                           description=eval_help,
                                           epilog=docs,
                                           help=eval_help)

        eval_parser.add_argument("psm_files",
                                 type=str,
                                 nargs="+",
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
                                 default="xenith",
                                 help=("The fileroot string will be added as a"
                                       " prefix to all output file names. This"
                                       " is the 'xenith' by default."))

        mod_help = """Which model should be used? Choices include several
        included pretrained models for output from Kojak, or custom models
        created in xenith or from Percolator. Specifically, the choices are:
        (1) 'kojak_mlp' - A multilayer perceptron model for output from
        Kojak 2.0.
        (2) 'kojak_linear' - A linear model for output from Kojak 2.0.
        (3) 'kojak_percolator' - A model created from Percolator results for
        output from Kojak 2.0.
        (4) 'custom' - A custom xenith model, created using 'xenith train'
        (5) 'custom_percolator' - A custom model created from the weights
        learned by Percolator. For the latter two, the '--model-file' option
        is required.
        """
        eval_parser.add_argument("--model",
                                 type=str,
                                 choices=["kojak_mlp", "kojak_linear",
                                          "kojak_percolator", "custom",
                                          "custom_percolator"],
                                 default="kojak_mlp",
                                 help=mod_help)

        eval_parser.add_argument("--model-file",
                                 type=str,
                                 help=("The path to a custom model. If "
                                       "'--model' is 'custom', this should "
                                       "be a xenith model file. If '--model' "
                                       "is 'custom_percolator' this should be "
                                       "the weights file output by "
                                       "Percolator"))


        # Train
        fit_help = ("Train a new model to assess future collections of "
                    "cross-linked peptide-spectrum matches (PSMs).")
        fit_parser = subparser.add_parser("fit",
                                          parents=[common],
                                          description=fit_help,
                                          epilog=docs,
                                          help=fit_help)

        fit_parser.add_argument("psm_files",
                                type=str,
                                nargs="+",
                                help=("One or more collections of PSMs in "
                                      "the xenith tab-separated format"))

        fit_parser.add_argument("--fileroot",
                                type=str,
                                default="xenith_model",
                                nargs=1,
                                help=("The fileroot string will be added as "
                                      "a prefix to all output file names. "
                                      "This is 'xenith_model' by default."))

        fit_parser.add_argument("--hidden_dims",
                                type=str,
                                default="8,8,8",
                                nargs=1,
                                help=("A comma-separated list indicating "
                                      "dimensions of hidden layers to use "
                                      "in the model. This is '8,8,8' by "
                                      "default, which defines three hidden "
                                      "layers with 8 neurons each. If '', a "
                                      "linear model is used."))

        fit_parser.add_argument("--max_epochs",
                                type=int,
                                default=100,
                                nargs=1,
                                help=("The maximum number of epochs "
                                      "used for model training."))

        fit_parser.add_argument("--learning_rate",
                                type=float,
                                default=0.001,
                                nargs=1,
                                help=("The learning rate to be used for "
                                        "optimization with Adam. The default "
                                        "is 0.001"))

        fit_parser.add_argument("--batch_size",
                                type=int,
                                default=1028,
                                nargs=1,
                                help=("The batch size to use for "
                                      "optimization with Adam. The default "
                                      "is 1028."))

        fit_parser.add_argument("--weight_decay",
                                type=float,
                                default=0.01,
                                nargs=1,
                                help=("Adds L2 regularization to all model "
                                      "parameters"))

        fit_parser.add_argument("--early_stop",
                                type=int,
                                default=5,
                                help=("Stop training if the validation set "
                                      "loss does not decreases for n "
                                      "consecutive epochs. 0 disables early "
                                      "stopping."))

        fit_parser.add_argument("--gpu",
                                type=bool,
                                default=False,
                                help="Should a GPU be used, if available?")

        fit_parser.add_argument("--seed",
                                type=int,
                                default=1,
                                help="Set the seed for reproducibility.")


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
