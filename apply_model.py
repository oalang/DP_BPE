import sys
from bpe import Vocabulary, Model


def main():
    model_fname = sys.argv[1]
    text_fname = sys.argv[2]
    subwrd_fname = sys.argv[3]

    model_file = open(model_fname, 'r')
    model = Model.from_file(model_file)
    model_file.close()

    text_file = open(text_fname, 'r')
    subwrd_file = open(subwrd_fname, 'w')
    vocab = Vocabulary()
    for line in text_file:
        mappings = []
        for word in line.split():
            if vocab.missing(word):
                vocab.add_word(word, model=model)
            mappings.append(vocab.map(word))
        subwrd_file.write(' '.join(mappings) + "\n")
    text_file.close()
    subwrd_file.close()


if __name__ == '__main__':
    main()
