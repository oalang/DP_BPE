import collections
import re
import sys


class Word:
    def __init__(self, word):
        self.word = word
        self.freq = 0
        self.subwords = ["_"] + list(word)
        self.bigrams = []
        for i in range(len(self.subwords) - 1):
            self.bigrams.append((self.subwords[i], self.subwords[i + 1]))

    def apply_model(self, model):
        subwords = self.subwords
        bigrams = self.bigrams
        for pair in model.operations:
            if pair in bigrams:
                i = 0
                while i < len(subwords) - 1:
                    if (subwords[i], subwords[i + 1]) == pair:
                        subwords[i] = pair[0] + pair[1]
                        subwords.pop(i + 1)
                    i += 1
                bigrams.clear()
                for i in range(len(subwords) - 1):
                    bigrams.append((subwords[i], subwords[i + 1]))

    def update_count(self, n):
        self.freq += n
        assert self.freq >= 0, "Frequency of '%s' is %d, which is less than 0" % (self.word, self.freq)

    def show(self):
        return self.word + " " + str(self.freq)

    def print(self):
        print(self.show())


class Bigram:
    def __init__(self, a, b):
        self.pair = (a, b)
        self.freq = 0

    def update_count(self, n):
        self.freq += n
        assert self.freq >= 0, "Frequency of %s is %d, which is less than 0" % (str(self.pair), self.freq)

    def print(self):
        print(str(self.pair) + " " + str(self.freq))


class Vocabulary:
    def __init__(self):
        self.dict = {}
        self.list = []
        self.sorted = False
        self.char_set = set()
        self.sbwd_set = set()

    @classmethod
    def from_text(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            line = re.sub(r"[^A-Z']", ' ', line)
            for word in line.split():
                if new_vocab.missing(word):
                    new_vocab.add_word(word)
                new_vocab.dict[word].update_count(1)
        new_vocab.sort()
        return new_vocab

    @classmethod
    def from_vocab(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            entry = line.split()
            word = entry[0]
            frequency = int(entry[1])
            new_vocab.add_word(word)
            new_vocab.dict[word].update_count(frequency)
        new_vocab.sort()
        return new_vocab

    def add_word(self, word, model=None):
        assert word not in self.dict, "'%s' already in vocabulary" % word
        new_word = Word(word)
        self.dict[word] = new_word
        self.list.append(new_word)
        self.char_set.update(new_word.subwords)
        if model is not None:
            new_word.apply_model(model)
        self.sbwd_set.update(new_word.subwords)

    def sort(self):
        s = sorted(self.list, key=lambda word: word.word)
        self.list = sorted(s, key=lambda word: word.freq, reverse=True)
        self.sorted = True

    def print(self, file=None, max_print=None):
        i = 0
        for word in self.list:
            if file is None:
                word.print()
            else:
                file.write(word.show() + "\n")
            i += 1
            if max_print is not None and i >= max_print:
                break

    def apply_operation(self, pair):
        updates = collections.defaultdict(int)
        for word in self.list:
            subwords = word.subwords
            bigrams = word.bigrams
            freq = word.freq
            if pair in bigrams:
                self.sbwd_set.update({pair[0] + pair[1]})
                i = 0
                while i < len(subwords) - 1:
                    if (subwords[i], subwords[i + 1]) == pair:
                        subwords[i] = pair[0] + pair[1]
                        subwords.pop(i + 1)
                        updates[pair] -= freq
                        if i > 0:
                            updates[(subwords[i - 1], pair[0])] -= freq
                            updates[(subwords[i - 1], subwords[i])] += freq
                        if i < len(subwords) - 1:
                            updates[(pair[1], subwords[i + 1])] -= freq
                            updates[(subwords[i], subwords[i + 1])] += freq
                    i += 1
                bigrams.clear()
                for i in range(len(subwords) - 1):
                    bigrams.append((subwords[i], subwords[i + 1]))
        return updates

    def missing(self, word):
        if word in self.dict:
            return False
        else:
            return True

    def map(self, word):
        subwords = self.dict[word].subwords
        return ' '.join(subwords)


class Statistics:
    def __init__(self):
        self.dict = {}
        self.list = []
        self.sorted = False

    @classmethod
    def from_vocab(cls, vocab):
        new_stats = cls()
        for word in vocab.dict:
            bigrams = vocab.dict[word].bigrams
            freq = vocab.dict[word].freq
            for pair in bigrams:
                new_stats.add_pair(pair)
                new_stats.dict[pair].update_count(freq)
        new_stats.sort()
        return new_stats

    def add_pair(self, pair):
        if pair not in self.dict:
            new_pair = Bigram(pair[0], pair[1])
            self.dict[pair] = new_pair
            self.list.append(new_pair)
            self.sorted = False

    def sort(self):
        s = sorted(self.list, key=lambda pair: pair.pair)
        self.list = sorted(s, key=lambda pair: pair.freq, reverse=True)
        self.sorted = True

    def max_pair(self):
        if not self.sorted:
            self.sort()
        if self.list and self.list[0].freq > 0:
            return self.list[0].pair
        else:
            return None

    def print(self, max_print=None):
        i = 0
        for pair in self.list:
            pair.print()
            i += 1
            if max_print is not None and i >= max_print:
                break

    def update(self, updates):
        for pair in updates:
            self.add_pair(pair)
            self.dict[pair].update_count(updates[pair])
        self.sorted = False


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
