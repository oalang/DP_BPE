"""
Groups subwords into words writes the result to an output file
"""

import argparse


class Arguments:
    def __init__(self, args):
        self.subwrd_fname = args.subwords
        self.text_fname = args.output

    def valid(self):
        return self.model_fname is not None and self.text_fname is not None and self.subwrd_fname is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--subwords",
                            help="file path for subword text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for text output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.subwrd_fname is None:
            message += "Subword file must be specified\n"
        if self.text_fname is None:
            message += "Output file must be specified\n"
        return message


def decode_bpe(args):
    subwrd_fname = args.subwrd_fname
    text_fname = args.text_fname

    # Group subwords into words and write the results to a file.
    with open(subwrd_fname, 'r') as text_file, open(text_fname, 'w') as subwrd_file:
        for line in subwrd_file:
            line.replace(" ", "").replace("_", " ").strip()
            text_file.write(line + "\n")


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        decode_bpe(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
