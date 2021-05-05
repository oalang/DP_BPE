"""
Trains a BPE model from a given vocabulary and saves it to an output file with the format:
<subword_A> <subword_B>
<subword_D> <subword_C>
...
"""

import argparse
import time
from bpe import Vocabulary, Statistics, Model


class Arguments:
    def __init__(self, args):
        self.vocab_fname = args.vocab
        self.model_fname = args.output
        self.max_subwords = args.max_subwords

    def valid(self):
        return self.vocab_fname is not None and self.model_fname is not None and self.max_subwords > 0

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--vocab",
                            help="file path to vocabulary to be trained on",
                            type=str)
        parser.add_argument("--output",
                            help="file path for model output",
                            type=str)
        parser.add_argument("--max-subwords",
                            help="maximum size of subword vocabulary",
                            type=int,
                            default=1000)
        return parser

    def invalid_opts(self):
        message = ""
        if self.vocab_fname is None:
            message += "Vocabulary file must be specified\n"
        if self.model_fname is None:
            message += "Output file must be specified\n"
        if self.max_subwords <= 0:
            message += "Max operations must be greater than 0\n"
        return message


def train_model(args):
    vocab_fname = args.vocab_fname
    model_fname = args.model_fname
    max_subwords = args.max_subwords

    with open(vocab_fname, 'r') as vocab_file:
        vocab = Vocabulary.from_vocab(vocab_file)

    max_operations = max_subwords - vocab.num_char()
    stats = Statistics.from_vocab(vocab)
    model = Model()
    for i in range(max_operations):
        max_pair = stats.max_pair()
        if max_pair is None:
            print(f"Stopped early with {i} operations")
            break
        model.add_operation(max_pair)
        updates = vocab.apply_operation(max_pair)
        stats.update(updates)

    with open(model_fname, 'w') as model_file:
        model.print(file=model_file)


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        start_time = time.time()
        train_model(args)
        print(time.time()-start_time)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
