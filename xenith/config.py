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
        subparser = parser.add_subparsers(title="Command", dest="command")

        # Predict
        predict_help = ("Use a previously trained model to assess a new"
                        " collection of peptide-spectrum matches (PSMs).")

        predict = subparser.add_parser("predict",
                                       parents=[common],
                                       description=predict_help,
                                       epilog=docs,
                                       help=predict_help)

        mod_help = """What model should me used? The easiest option is to
        select one of the xenith pretrained models designed for output from
        Kojak. These are:
        (1) 'kojak_mlp' - A multilayer perceptron model for output from
        Kojak 2.0.
        (2) 'kojak_linear' - A linear model for output from Kojak 2.0.
        (3) 'kojak_percolator' - A model created from Percolator results for
        output from Kojak 2.0.
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
                                   " is the 'xenith' by default."))

        # Kojak conversion
        kojak_help = ("Convert Kojak search results to the xenith tab-"
                      "delimited format.")

        kojak = subparser.add_parser("kojak",
                                     parents=[common],
                                     description=kojak_help,
                                     epilog=docs,
                                     help=kojak_help)

        kojak.add_argument("kojak",
                           type=str,
                           help=("The path to the main kojak result file"
                                 " (*.kojak.txt)."))

        kojak.add_argument("perc-intra",
                           type=str,
                           help=("The path to the intraprotein Percolator "
                                 "input file from Kojak (*.perc.intra.txt)."))

        kojak.add_argument("perc-inter",
                           type=str,
                           help=("The path to the interprotein Percolator "
                                 "input file from Kojak (*.perc.inter.txt)."))

        kojak.add_argument("-o", "--output-file",
                           type=str,
                           default="kojak.xenith.txt",
                           help=("The output file name and path."))

        kojak.add_argument("-c", "--max-charge",
                           type=int,
                           default=8,
                           help="The maximum charge state to consider.")

        kojak.add_argument("-p", "--to-pin",
                           type=bool,
                           default=False,
                           help=("Convert to a Percolator INput file instead "
                                 "of the xenith tab-delimited format."))

        self._namespace = vars(parser.parse_args())
        self.parser = parser

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
