"""
Classes and methods for generating a vocabulary, training a BPE model, and applying it to text
"""

from collections import defaultdict
import re


# Word object contains a token, its frequency in the vocabulary, and its current subword mapping.
class Word:
    def __init__(self, token):
        self.token = token
        self.freq = 0
        # Begin with individual characters plus an end line symbol.
        self.subwords = list(token) + ["_"]

    def update_freq(self, n):
        self.freq += n
        assert self.freq >= 0, f"Frequency of '{self.token}' is {self.freq}, which is less than 0"

    def apply_model(self, model):
        # Run through each subword grouping operation in order and apply it to the subword mapping.
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


# Bigram object contains a subword pair, its current overall frequency in the vocabulary,
# and a dictionary of tokens which currently contain the bigram in their subword mappings.
# Keeping track of which tokens contain a bigram results in a significant time reduction
# when removing a bigram from the vocabulary.
class Bigram:
    def __init__(self, pair):
        self.pair = pair
        self.freq = 0
        self.token_freq = defaultdict(int)

    def update_token_freq(self, token_updates):
        # Update the frequency of the bigram in every token where it has changed.
        for token, n in token_updates.items():
            self.token_freq[token] += n
            # Remove a token from the frequency dictionary if it no longer contains the bigram
            if not self.token_freq[token]:
                del self.token_freq[token]
            self.freq += n
        assert self.freq >= 0, f"Frequency of {self.pair} is {self.freq}, which is less than 0"


# Vocabulary object contains a dictionary of all the tokens appearing in a given text
# and matches them to corresponding Word instances with up-to-date subwords mappings.
# It also keeps a set of all the characters found in its tokens.
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
        # When a BPE model is being applied to a text, a subword mapping is generated for each token
        # when it is first encountered and added to the vocabulary.
        if model:
            new_word.apply_model(model)

    def num_char(self):
        return len(self.char_set)

    def replace_bigram(self, bigram):
        # For every token which contains the bigram in its current subword mapping,
        # replace the bigram by grouping its elements into one. Keep track of which other
        # bigrams are lost and gained in each token's subword mapping and produce a
        # dictionary of update dictionaries for each bigram.
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

    def write(self, file):
        for word in sorted(self.tokn_dict.values(), key=lambda x: (-x.freq, x.token)):
            file.write(f"{word.token} {word.freq}\n")


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
            # Remove a bigram from the bigram dictionary if it no longer appears in any subword mappings
            if not self.bgrm_dict[pair].freq:
                del self.bgrm_dict[pair]

    def max_bigram(self):
        return max(self.bgrm_dict.values(), key=lambda bigram: bigram.freq, default=None)


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

    def write(self, file):
        for operation in self.operations:
            file.write(f"{operation[0]} {operation[1]}\n")
