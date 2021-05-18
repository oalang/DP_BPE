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
                            help="file path for bpe model",
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
        if self.model_fname is None:
            message += "Model file must be specified\n"
        if self.text_fname is None:
            message += "Text file must be specified\n"
        if self.subwrd_fname is None:
            message += "Output file must be specified\n"
        return message


def encode_text(args):
    model_fname = args.model_fname
    text_fname = args.text_fname
    subwrd_fname = args.subwrd_fname

    # Load the BPE model from a file.
    with open(model_fname, 'r') as model_file:
        model = Model.from_model_file(model_file)

    # Break up each word in a text file into subwords. The first time a word a is seen, a subword
    # mapping is generated by applying the model and added to a vocabulary. Subsequent instances are
    # looked up in the vocabulary. Write the results to a file.
    with open(text_fname, 'r') as text_file, open(subwrd_fname, 'w') as subwrd_file:
        vocab = Vocabulary()
        for line in text_file:
            mappings = []
            for token in line.split():
                if vocab.missing(token):
                    vocab.add_word(token, model)
                mappings.append(vocab.map_to_sbwds(token))
            subwrd_file.write(' '.join(mappings) + "\n")


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        encode_text(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
