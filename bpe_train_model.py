#!/usr/bin/python3

"""
Trains a BPE model from a given vocabulary and saves it to an output file.

Output file format::

    <subword_A> <subword_B>
    <subword_C> <subword_D>
    ...

Example::

    bpe_train_model.py --vocabulary vocabulary.txt --max-subwords 100 --output bpe_model.txt
"""

import argparse

from bpe import Vocabulary, train_model


class Arguments:
    def __init__(self, args):
        self.vocabulary_path = args.vocabulary
        self.bpe_model_path = args.output
        self.max_subwords = args.max_subwords

    def valid(self):
        return self.vocabulary_path is not None and self.bpe_model_path is not None and self.max_subwords > 0

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument("--vocabulary",
                            help="file path for vocabulary to be trained on",
                            type=str)
        parser.add_argument("--output",
                            help="file path for BPE model output",
                            type=str)
        parser.add_argument("--max-subwords",
                            help="maximum size of subword vocabulary",
                            type=int,
                            default=1000)
        return parser

    def invalid_opts(self):
        message = ""
        if self.vocabulary_path is None:
            message += "Vocabulary file must be specified\n"
        if self.bpe_model_path is None:
            message += "Output file must be specified\n"
        if self.max_subwords <= 0:
            message += "Max subwords must be greater than 0\n"
        return message


def bpe_train_model(args):
    vocabulary_path = args.vocabulary_path
    bpe_model_path = args.bpe_model_path
    max_subwords = args.max_subwords

    with open(vocabulary_path, 'r') as vocabulary_file:
        vocabulary = Vocabulary.from_vocabulary_file(vocabulary_file)

    bpe_model = train_model(vocabulary, max_subwords)

    with open(bpe_model_path, 'w') as bpe_model_file:
        bpe_model.write(file=bpe_model_file)


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        bpe_train_model(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
