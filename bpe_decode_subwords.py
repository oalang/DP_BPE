#!/usr/bin/python3

"""
Groups subwords into words and writes the result to an output file.

Example:
    bpe_decode_subwords.py --subwords subwords.txt --output text.txt
"""

import argparse

from bpe import decode_subwords


class Arguments:
    def __init__(self, args):
        self.subword_path = args.subwords
        self.text_path = args.output

    def valid(self):
        return self.subword_path is not None and self.text_path is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--subwords",
                            help="file path for subword text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for text output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.subword_path is None:
            message += "Subword file must be specified\n"
        if self.text_path is None:
            message += "Output file must be specified\n"
        return message


def bpe_decode_subwords(args):
    subword_path = args.subword_path
    text_path = args.text_path

    with open(subword_path, 'r') as subword_file, open(text_path, 'w') as text_file:
        for line in subword_file:
            text = decode_subwords(line)
            text_file.write(text + "\n")


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        bpe_decode_subwords(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
