import collections
import re
import sys


class Word:
    def __init__(self, token):
        self.token = token
        self.freq = 0
        self.subwords = list(token) + ["_"]

    def apply_model(self, model):
        subwords = self.subwords
        operations = model.operations
        for pair in operations:
            a, b = pair
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    subword = a + b
                    subwords[i] = subword
                    subwords.pop(i + 1)
                i += 1

    def update_count(self, n):
        self.freq += n
        assert self.freq >= 0, f"Frequency of {self.token} is {self.freq}, which is less than 0"

    def show(self):
        return self.token + " " + str(self.freq)

    def print(self):
        print(self.show())


class Bigram:
    def __init__(self, pair):
        self.pair = pair
        self.freq = 0

    def update_count(self, n):
        self.freq += n
        assert self.freq >= 0, f"Frequency of {str(self.pair)} is {self.freq}, which is less than 0"

    def show(self):
        return str(self.pair) + " " + str(self.freq)

    def print(self):
        print(str(self.show()))


class Vocabulary:
    def __init__(self):
        self.tokn_dict = {}
        self.char_set = set()
        self.sbwd_set = set()

    @classmethod
    def from_text(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            line = re.sub(r"[^A-Z']", ' ', line)
            for token in line.split():
                if new_vocab.missing(token):
                    new_vocab.add_word(token)
                new_vocab.tokn_dict[token].update_count(1)
        return new_vocab

    @classmethod
    def from_vocab(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            entry = line.split()
            token = entry[0]
            freq = int(entry[1])
            new_vocab.add_word(token)
            new_vocab.tokn_dict[token].update_count(freq)
        return new_vocab

    def missing(self, token):
        if token in self.tokn_dict:
            return False
        else:
            return True

    def add_word(self, token, model=None):
        assert token not in self.tokn_dict, f"'{token}' already in token dictionary"
        new_word = Word(token)
        self.tokn_dict[token] = new_word
        self.char_set.update(new_word.subwords)
        if model is not None:
            new_word.apply_model(model)
        self.sbwd_set.update(new_word.subwords)

    def apply_operation(self, pair):
        a, b = pair
        updates = collections.defaultdict(int)
        for word in self.tokn_dict.values():
            subwords = word.subwords
            freq = word.freq
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    subword = a + b
                    subwords[i] = subword
                    subwords.pop(i + 1)
                    updates[(a, b)] -= freq
                    if i > 0:
                        updates[(subwords[i - 1], a)] -= freq
                        updates[(subwords[i - 1], subwords[i])] += freq
                    if i < len(subwords) - 1:
                        updates[(b, subwords[i + 1])] -= freq
                        updates[(subwords[i], subwords[i + 1])] += freq
                    self.sbwd_set.update({subword})
                i += 1
        return updates

    def map_to_sbwds(self, token):
        subwords = self.tokn_dict[token].subwords
        return ' '.join(subwords)

    def sorted(self):
        words = sorted(self.tokn_dict.values(), key=lambda word: word.token)
        words = sorted(words, key=lambda word: word.freq, reverse=True)
        return words

    def print(self, file=None, max_print=None):
        i = 0
        for word in self.sorted():
            if file is None:
                word.print()
            else:
                file.write(word.show() + "\n")
            i += 1
            if max_print is not None and i >= max_print:
                break


class Statistics:
    def __init__(self):
        self.bgrm_dict = {}

    @classmethod
    def from_vocab(cls, vocab):
        new_stats = cls()
        for token in vocab.tokn_dict:
            subwords = vocab.tokn_dict[token].subwords
            freq = vocab.tokn_dict[token].freq
            for i in range(len(subwords)-1):
                pair = (subwords[i], subwords[i + 1])
                if new_stats.missing(pair):
                    new_stats.add_bigram(pair)
                new_stats.bgrm_dict[pair].update_count(freq)
        return new_stats

    def missing(self, pair):
        if pair in self.bgrm_dict:
            return False
        else:
            return True

    def add_bigram(self, pair):
        assert pair not in self.bgrm_dict, f"{str(pair)} already in bigram dictionary"
        new_bigram = Bigram(pair)
        self.bgrm_dict[pair] = new_bigram

    def max_pair(self):
        max_bigram = max(self.bgrm_dict.values(), key=lambda bigram: bigram.freq, default=None)
        if max_bigram is None:
            return None
        else:
            return max_bigram.pair

    def update(self, updates):
        for pair in updates:
            if self.missing(pair):
                self.add_bigram(pair)
            self.bgrm_dict[pair].update_count(updates[pair])
            if self.bgrm_dict[pair].freq == 0:
                del self.bgrm_dict[pair]

    def sorted(self):
        bigrams = sorted(self.bgrm_dict.values(), key=lambda bigram: bigram.pair)
        bigrams = sorted(bigrams, key=lambda bigram: bigram.freq, reverse=True)
        return bigrams

    def print(self, file=None, max_print=None):
        i = 0
        for bigram in self.sorted():
            if file is None:
                bigram.print()
            else:
                file.write(bigram.show() + "\n")
            i += 1
            if max_print is not None and i >= max_print:
                break


class Model:
    def __init__(self):
        self.operations = []

    @classmethod
    def from_file(cls, file):
        new_model = cls()
        for line in file:
            entry = line.split()
            new_model.add((entry[0], entry[1]))
        return new_model

    def add(self, pair):
        self.operations.append(pair)

    def print(self, file=None, max_print=None):
        i = 0
        for operation in self.operations:
            if file is None:
                print(operation)
            else:
                file.write(operation[0] + " " + operation[1] + "\n")
            i += 1
            if max_print is not None and i >= max_print:
                break


def main():
    file = open(sys.argv[1])
    vocab = Vocabulary.from_text(file)
    stats = Statistics.from_vocab(vocab)
    model = Model()
    vocab.print(max_print=10)
    stats.print(max_print=10)
    for i in range(1000):
        best = stats.max_pair()
        if best is None:
            break
        model.add(best)
        updates = vocab.apply_operation(best)
        stats.update(updates)
    model.print(max_print=10)

    file.close()


if __name__ == '__main__':
    main()
