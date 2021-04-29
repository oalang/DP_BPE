import sys
from bpe import Vocabulary


def main():
    text_fname = sys.argv[1]
    vocab_fname = sys.argv[2]

    text_file = open(text_fname, 'r')
    vocab = Vocabulary.from_text(text_file)
    text_file.close()

    vocab_file = open(vocab_fname, 'w')
    vocab.print(file=vocab_file)
    vocab_file.close()


if __name__ == '__main__':
    main()
