import sys
from bpe import Vocabulary, Statistics, Model


def main():
    vocab_fname = sys.argv[1]
    model_fname = sys.argv[2]
    max_operations = 5000
    if len(sys.argv) > 3:
        max_operations = int(sys.argv[3])

    vocab_file = open(vocab_fname, 'r')
    vocab = Vocabulary.from_vocab(vocab_file)
    vocab_file.close()

    stats = Statistics.from_vocab(vocab)
    model = Model()
    for i in range(max_operations):
        best = stats.max_pair()
        if best is None:
            break
        model.add(best)
        updates = vocab.apply_operation(best)
        stats.update(updates)

    model_file = open(model_fname, 'w')
    model.print(file=model_file)
    model_file.close()


if __name__ == '__main__':
    main()
