"""
Uses a BPE model to breakup a text file into subwords and writes the result to an output file
"""

import argparse
from bpe import Vocabulary, Model


class Arguments:
    def __init__(self, args):
        self.model_fname = args.model
        self.text_fname = args.text
        self.subwrd_fname = args.output

    def valid(self):
        return self.model_fname is not None and self.text_fname is not None and self.subwrd_fname is not None

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model",
                            help="file path to bpe model",
                            type=str)
        parser.add_argument("--text",
                            help="file path to text being processed",
                            type=str)
        parser.add_argument("--output",
                            help="file path for subword output",
                            type=str)
        return parser

    def invalid_opts(self):
        message = ""
        if self.model_fname is None:
            message += "Model file must be specified\n"
        if self.text_fname is None:
            message += "Text file must be specified\n"
        if self.subwrd_fname is None:
            message += "Output file must be specified\n"
        return message


def apply_model(args):
    model_fname = args.model_fname
    text_fname = args.text_fname
    subwrd_fname = args.subwrd_fname

    model_file = open(model_fname, 'r')
    model = Model.from_file(model_file)
    model_file.close()

    text_file = open(text_fname, 'r')
    subwrd_file = open(subwrd_fname, 'w')
    vocab = Vocabulary()
    for line in text_file:
        mappings = []
        for token in line.split():
            if vocab.missing(token):
                vocab.add_word(token, model=model)
            mappings.append(vocab.map_to_sbwds(token))
        subwrd_file.write(' '.join(mappings) + "\n")
    text_file.close()
    subwrd_file.close()


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        apply_model(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
