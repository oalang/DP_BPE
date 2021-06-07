"""
Trains a BPE model from a given vocabulary and saves it to an output file with the format:
<subword_A> <subword_B>
<subword_D> <subword_C>
...
"""

import argparse

from bpe import Vocabulary, Statistics, BPEModel


class Arguments:
    def __init__(self, args):
        self.vocabulary_fname = args.vocabulary
        self.bpe_model_fname = args.output
        self.max_subwords = args.max_subwords

    def valid(self):
        return self.vocabulary_fname is not None and self.bpe_model_fname is not None and self.max_subwords > 0

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description=__doc__)
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
        if self.vocabulary_fname is None:
            message += "Vocabulary file must be specified\n"
        if self.bpe_model_fname is None:
            message += "Output file must be specified\n"
        if self.max_subwords <= 0:
            message += "Max subwords must be greater than 0\n"
        return message


def train_bpe_model(args):
    vocabulary_fname = args.vocabulary_fname
    bpe_model_fname = args.bpe_model_fname
    max_subwords = args.max_subwords

    # Load the vocabulary from a file.
    with open(vocabulary_fname, 'r') as vocabulary_file:
        vocabulary = Vocabulary.from_vocabulary_file(vocabulary_file)

    # Subtract the number characters in the vocabulary from the maximum number of concatenation operations.
    max_operations = max_subwords - vocabulary.num_characters()

    # Compile initial statistics from the vocabulary.
    statistics = Statistics.from_vocabulary(vocabulary)

    # Train a BPE model by iteratively adding the most frequent bigram to the model, replacing that bigram
    # in the vocabulary's subword mappings with its concatenation, removing it from the statistics, and
    # updating the frequencies of other bigrams affected by the operation.
    bpe_model = BPEModel()
    for i in range(max_operations):
        max_bigram = statistics.max_bigram()
        if max_bigram is None:
            print(f"Stopped early with {i} operations")
            break
        bpe_model.add_operation(max_bigram)
        bigram_updates = vocabulary.replace_bigram(max_bigram)
        statistics.remove_bigram(max_bigram)
        statistics.update_bigram_frequencies(bigram_updates)

    # Write the bpe_model to a file.
    with open(bpe_model_fname, 'w') as bpe_model_file:
        bpe_model.write(file=bpe_model_file)


def main():
    parser = Arguments.get_parser()
    args = Arguments(parser.parse_args())
    if args.valid():
        train_bpe_model(args)
    else:
        print("Error: Invalid Options\n" + args.invalid_opts())


if __name__ == '__main__':
    main()
