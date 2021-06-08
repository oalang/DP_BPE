#!/usr/bin/python3

"""
Uses a BPE model to breakup a text file into subwords and writes the result to an output file.

Example:
    bpe_encode_text.py --bpe-model bpe_model.txt --text sample_text.txt --output subwords.txt
"""

import argparse

from bpe import Vocabulary, BPEModel, encode_text


class Arguments:
    def __init__(self, args):
        self.bpe_model_path = args.bpe_model
        self.text_path = args.text
        self.subword_path = args.output

    def valid(self):
        return self.bpe_model_path is not None and self.text_path is not None and self.subword_path is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bpe-model",
                            help="file path for BPE model",
                            type=str)
        parser.add_argument("--text",
                            help="file path for text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for subword output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.bpe_model_path is None:
            message += "BPE model file must be specified\n"
        if self.text_path is None:
            message += "Text file must be specified\n"
        if self.subword_path is None:
            message += "Output file must be specified\n"
        return message


def bpe_encode_text(args):
    bpe_model_path = args.bpe_model_path
    text_path = args.text_path
    subword_path = args.subword_path

    with open(bpe_model_path, 'r') as bpe_model_file:
        bpe_model = BPEModel.from_model_file(bpe_model_file)

    with open(text_path, 'r') as text_file, open(subword_path, 'w') as subword_file:
        vocabulary = Vocabulary()
        for line in text_file:
            subwords = encode_text(line, bpe_model, vocabulary)
            subword_file.write(subwords + "\n")


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        bpe_encode_text(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
