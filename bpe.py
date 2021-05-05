"""
Classes and methods for generating a vocabulary, training a BPE model, and applying it to text
"""

from collections import defaultdict
import re


class Word:
    def __init__(self, token):
        self.token = token
        self.freq = 0
        self.subwords = list(token) + ["_"]

    def update_freq(self, n):
        self.freq += n
        assert self.freq >= 0, f"Frequency of '{self.token}' is {self.freq}, which is less than 0"

    def show(self):
        return f"{self.token} {self.freq}"

    def print(self):
        print(self.show())

    def apply_model(self, model):
        subwords = self.subwords
        operations = model.operations
        for pair in operations:
            a, b = pair
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    subwords[i] = a + b
                    del subwords[i + 1]
                i += 1


class Bigram:
    def __init__(self, pair):
        self.pair = pair
        self.freq = 0
        self.token_freq = defaultdict(int)

    def update_token_freq(self, token_updates):
        for token, n in token_updates.items():
            self.token_freq[token] += n
            if not self.token_freq[token]:
                del self.token_freq[token]
            self.freq += n
        assert self.freq >= 0, f"Frequency of {self.pair} is {self.freq}, which is less than 0"

    def show(self):
        return f"{self.pair} {self.freq}"

    def print(self):
        print(str(self.show()))


class Vocabulary:
    def __init__(self):
        self.tokn_dict = {}
        self.char_set = set()

    @classmethod
    def from_text_file(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            line = re.sub(r"[^A-Z']", " ", line)
            for token in line.split():
                if new_vocab.missing(token):
                    new_vocab.add_word(token)
                new_vocab.tokn_dict[token].update_freq(1)
        return new_vocab

    @classmethod
    def from_vocab_file(cls, file):
        new_vocab = cls()
        for line in file:
            line = line.upper()
            entry = line.split()
            token = entry[0]
            freq = int(entry[1])
            new_vocab.add_word(token)
            new_vocab.tokn_dict[token].update_freq(freq)
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

    def num_char(self):
        return len(self.char_set)

    def replace_bigram(self, bigram):
        a, b = bigram.pair
        tokens = bigram.token_freq.keys()
        updates = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            freq = self.tokn_dict[token].freq
            subwords = self.tokn_dict[token].subwords
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    subwords[i] = a + b
                    del subwords[i + 1]
                    updates[(a, b)][token] -= freq
                    if i > 0:
                        updates[(subwords[i - 1], a)][token] -= freq
                        updates[(subwords[i - 1], subwords[i])][token] += freq
                    if i < len(subwords) - 1:
                        updates[(b, subwords[i + 1])][token] -= freq
                        updates[(subwords[i], subwords[i + 1])][token] += freq
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
                new_stats.bgrm_dict[pair].update_token_freq({token: freq})
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

    def update_bigrams(self, bigram_updates):
        for pair, token_updates in bigram_updates.items():
            if self.missing(pair):
                self.add_bigram(pair)
            self.bgrm_dict[pair].update_token_freq(token_updates)
            if not self.bgrm_dict[pair].freq:
                del self.bgrm_dict[pair]

    def max_bigram(self):
        max_bigram = max(self.bgrm_dict.values(), key=lambda bigram: bigram.freq, default=None)
        if max_bigram is None:
            return None
        else:
            return max_bigram

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
    def from_model_file(cls, file):
        new_model = cls()
        for line in file:
            line = line.upper()
            entry = line.split()
            new_model.add_operation((entry[0], entry[1]))
        return new_model

    def add_operation(self, pair):
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
