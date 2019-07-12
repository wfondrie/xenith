"""
Contains all the the configuration details for running xenith from the
command-line.
"""
import argparse
import textwrap


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

        docs = ("\nSee the official documentation website for more detailed "
                "documentation, examples, and best practices: "
                "https://xenith.readthedocs.com")

        # Setup main parser
        parser = argparse.ArgumentParser(description=desc + "\n \n",
                                         epilog=docs,
                                         formatter_class=XenithHelpFormatter)

        # Setup parser for shared arguments
        common = argparse.ArgumentParser(add_help=False)

        common.add_argument("-v", "--verbosity",
                            default=2,
                            type=int,
                            choices=[0, 1, 2, 3],
                            help=("Specify the verbosity of the current "
                                  "process. Each level prints the following "
                                  "messages, including all those at a lower "
                                  "verbosity: 0-errors, 1-warnings, 2-messages"
                                  ", 3-debug info."))

        # Add subparsers for each subcommand
        commands = parser.add_subparsers(title="COMMANDS", dest="command",
                                         help=("Available commands to run "
                                               "xenith and convert search "
                                               "engine results to xenith tab-"
                                               "delimited format."))

        # Predict
        predict_help = ("Use a previously trained model to assess a new"
                        " collection of peptide-spectrum matches (PSMs).")

        predict = commands.add_parser("predict",
                                      parents=[common],
                                      description=predict_help,
                                      epilog=docs,
                                      help=predict_help)

        mod_help = """What model should be used? The easiest option is to
        select one of the xenith pretrained models designed for output from
        Kojak. These are:
        (1) 'kojak_mlp' - A multilayer perceptron model for output from
        Kojak 2.0.0.
        (2) 'kojak_linear' - A linear model for output from Kojak 2.0.0.
        (3) 'kojak_percolator' - A model created from Percolator results from
        Kojak 2.0.0.
        (4) 'kojak_1.6.1_mlp' - A multilayer perceptron model for output from
        Kojak 1.6.1.
        Alternatively, a custom xenith model file or the weights output from
        Percolator can be used.
        """
        predict.add_argument("-m", "--model",
                             type=str,
                             default="kojak_mlp",
                             help=mod_help)

        predict.add_argument("psm_files",
                             type=str,
                             nargs="+",
                             help=("A collection of cross-linked PSMs in"
                                   " the xenith tab-delimited format."))

        predict.add_argument("-o", "--output_dir",
                             default=".",
                             type=str,
                             help=("The directory in which to write xenith "
                                   "results. This is the working directory by "
                                   "default."))

        predict.add_argument("-r", "--fileroot",
                             type=str,
                             default="xenith",
                             help=("The fileroot string will be added as a"
                                   " prefix to all output file names. This"
                                   " is the 'xenith' by default. (string)"))

        predict.add_argument("-s", "--seed",
                             type=int,
                             default=1,
                             help=("Integer indicating the random seed. "
                                   "Because tied PSMs are broken randomly,"
                                   "this ensures reproducibility."))


        # Kojak conversion
        kojak_help = ("Convert Kojak search results to the xenith tab-"
                      "delimited format.")

        kojak = commands.add_parser("kojak",
                                    parents=[common],
                                    description=kojak_help,
                                    epilog=docs,
                                    help=kojak_help)

        kojak.add_argument("kojak",
                           type=str,
                           help=("The path to the main kojak result file"
                                 " (*.kojak.txt)."))

        kojak.add_argument("perc_inter",
                           type=str,
                           help=("The path to the interprotein Percolator "
                                 "input file from Kojak (*.perc.inter.txt)."))

        kojak.add_argument("perc_intra",
                           type=str,
                           help=("The path to the intraprotein Percolator "
                                 "input file from Kojak (*.perc.intra.txt)."))

        kojak.add_argument("-o", "--output_file",
                           type=str,
                           default="kojak.xenith.txt",
                           help=("The output file name and path."))

        kojak.add_argument("-r", "--version",
                           type=str,
                           choices=["2.0-dev"],
                           default="2.0-dev",
                           help=("The version of Kojak that was used."))

        kojak.add_argument("-c", "--max_charge",
                           type=int,
                           default=8,
                           help=("Integer indicating the maximum charge state "
                                 "to consider. The default is 8."))

        kojak.add_argument("-d", "--decoy_prefix",
                           type=str,
                           default="decoy_",
                           help=("The prefix used to indicate decoy sequences."
                                 " The default is 'decoy_'"))

        kojak.add_argument("-p", "--to_pin",
                           type=bool,
                           default=False,
                           help=("Boolean indicating whether to convert Kojak "
                                 "results to a Percolator INput file instead "
                                 "of the xenith tab-delimited format. The "
                                 "default is False."))

        self._namespace = vars(parser.parse_args())
        self.parser = parser

    def __getattr__(self, option):
        return self._namespace[option]


# Utility Functions -----------------------------------------------------------
def _process_line(line, width, indent):
    line = textwrap.fill(line, width, initial_indent=indent,
                         subsequent_indent=indent,
                         replace_whitespace=False)
    return line.strip()
