"""
Extracts word tokens and frequencies from text and saves them to an output file with the format:
<word_token_1> <frequency_1>
<word_token_2> <frequency_2>
...
"""

import argparse

from bpe import Vocabulary


class Arguments:
    def __init__(self, args):
        self.text_fname = args.text
        self.vocabulary_fname = args.output

    def valid(self):
        return self.text_fname is not None and self.vocabulary_fname is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--text",
                            help="file path for text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for vocabulary output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.text_fname is None:
            message += "Text file must be specified\n"
        if self.vocabulary_fname is None:
            message += "Output file must be specified\n"
        return message


def compile_vocabulary(args):
    text_fname = args.text_fname
    vocabulary_fname = args.vocabulary_fname

    # Compile a vocabulary from a text file.
    with open(text_fname, 'r') as text_file:
        vocabulary = Vocabulary.from_text_file(text_file)

    # Write the vocabulary to a file.
    with open(vocabulary_fname, 'w') as vocabulary_file:
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
