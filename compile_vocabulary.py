#!/usr/bin/python3

"""
Extracts word tokens and frequencies from text and saves them to an output file.

Output file format::

    <word_token_1> <frequency_1>
    <word_token_2> <frequency_2>
    ...

Example::

    compile_vocabulary.py --text sample_text.txt --output vocabulary.txt
"""

import argparse

from bpe import Vocabulary


class Arguments:
    def __init__(self, args):
        self.text_path = args.text
        self.vocabulary_path = args.output

    def valid(self):
        return self.text_path is not None and self.vocabulary_path is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("--text",
                            help="file path for text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for vocabulary output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.text_path is None:
            message += "Text file must be specified\n"
        if self.vocabulary_path is None:
            message += "Output file must be specified\n"
        return message


def compile_vocabulary(args):
    text_path = args.text_path
    vocabulary_path = args.vocabulary_path

    with open(text_path, 'r') as text_file:
        vocabulary = Vocabulary.from_text_file(text_file)

    with open(vocabulary_path, 'w') as vocabulary_file:
        vocabulary.write(file=vocabulary_file)


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        compile_vocabulary(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
