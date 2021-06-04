"""
Classes and methods for generating a vocabulary, training a BPE model, and applying it to text
"""

from __future__ import annotations
from typing import Optional, TextIO
from collections import defaultdict
from math import ceil
import re


# This search set size was found to produce speedy results.
SEARCH_SET_TARGET_SIZE = 100


# Word object contains a token, its frequency in the vocabulary, and its current subword mapping.
class Word:
    def __init__(self, token: str) -> None:
        """
        Initialize a Word instance.

        The new instance will have an initial frequency of zero and a subword string
        split into individual characters followed by the end-of-token symbol '_'.

        Args:
            token: A unique token string used to identify this Word
        """

        self.token = token
        self.freq = 0
        self.subwords = list(token) + ['_']

    def update_freq(self, n: int) -> None:
        """
        Add n to the Word's frequency.

        Args:
            n: The number to add to the Word's frequency
        """

        self.freq += n

    def apply_model(self, model: BPEModel) -> None:
        """
        Apply a BPEModel to the Word to arrange its characters into subwords.

        Runs through each subword concatenation operation in order and applies it to
        the subword mapping.

        Args:
            model: The BPEModel to be used
        """

        subwords = self.subwords
        operations = model.operations
        for pair in operations:
            a, b = pair
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    # Replace target bigram.
                    subwords[i] = a + b
                    del subwords[i + 1]
                i += 1


# Bigram object contains a subword pair, its current overall frequency in the vocabulary, a dictionary
# of tokens which currently contain the bigram in their subword mappings, and a boolean indicating
# whether or not the bigram is in the current search set for most frequent bigram. Keeping track of
# which tokens contain a bigram results in a significant time reduction when removing a bigram from
# the vocabulary, compared to processing every token in the vocabulary.
class Bigram:
    def __init__(self, pair: tuple) -> None:
        """
        Initialize a Bigram instance.

        The new instance will have an initial frequency of zero.

        Args:
            pair: A pair of subword strings
        """

        self.pair = pair
        self.freq = 0
        self.token_freq = defaultdict(int)
        self.in_search_set = False

    def update_token_frequencies(self, token_updates: dict) -> None:
        """
        Update the Bigram's frequency statistics.

        Updates the frequency of the Bigram for every Word where it has changed, and
        also its overall frequency.

        Args:
            token_updates: A dictionary whose keys are tokens whose Word contains the
            subword pair and whose values are the updates
        """

        for token, n in token_updates.items():
            self.token_freq[token] += n
            # Remove a token from the frequency dictionary if it no longer contains the bigram.
            if self.token_freq[token] == 0:
                del self.token_freq[token]
            self.freq += n

    def add_to_search_set(self, search_set: set) -> None:
        """
        Add the Bigram to the search set.

        Adds the Bigram to the search set and sets its in_search_set flag to True.

        Args:
            search_set: The search set
        """

        search_set.add(self)
        self.in_search_set = True

    def remove_from_search_set(self, search_set: set) -> None:
        """
        Remove the Bigram from the search set.

        Removes the Bigram from the search set and sets its in_search_set flag to False.

        Args:
            search_set: The search set
        """

        search_set.remove(self)
        self.in_search_set = False


# Vocabulary object contains a dictionary of all the tokens appearing in a given text
# and matches them to corresponding Word instances with up-to-date subwords mappings.
# It also keeps a set of all the characters found in its tokens.
class Vocabulary:
    def __init__(self) -> None:
        """
        Initialize an empty Vocabulary instance.
        """

        self.tokn_dict = {}
        self.char_set = set()

    @classmethod
    def from_text_file(cls, file: TextIO) -> Vocabulary:
        """
        Initialize a Vocabulary from a text file.

        Normalizes the text by removing punctuation and capitalizing everything. Generates
        a new Word instance for each unique token string and counts its occurrences.

        Args:
            file: A file stream containing the text being processed

        Returns:
            A Vocabulary instance
        """

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
    def from_vocab_file(cls, file: TextIO) -> Vocabulary:
        """
        Initialize a Vocabulary from a vocabulary file.

        Reads in a vocabulary file with each line containing a unique token string followed
        by its frequency in the training set. For example, "THE 4827".

        Args:
            file: A file stream containing the vocabulary being processed

        Returns:
            A Vocabulary instance
        """

        new_vocab = cls()
        for line in file:
            line = line.upper()
            token, freq = line.split()
            freq = int(freq)
            new_vocab.add_word(token)
            new_vocab.tokn_dict[token].update_freq(freq)
        return new_vocab

    def missing(self, token: str) -> bool:
        """
        Check if a token string is missing from the token dictionary.

        Args:
            token: The token string to be searched for

        Returns:
            True if the token string is missing from the token dictionary and False otherwise
        """

        if token in self.tokn_dict:
            return False
        else:
            return True

    def add_word(self, token: str, model: Optional[BPEModel] = None) -> None:
        """
        Add a Word to the token dictionary.

        If a BPEModel is provided, it is applied to the Word to produce a subword mapping.

        Args:
            token: A token string representing the Word to be added
            model: A BPEModel used produce a subword mapping
        """

        new_word = Word(token)
        self.tokn_dict[token] = new_word
        self.char_set.update(new_word.subwords)
        if model:
            new_word.apply_model(model)

    def num_char(self) -> int:
        """
        Return the size of the Vocabulary's character set.

        Returns:
            The size of the Vocabulary's character set
        """

        return len(self.char_set)

    def replace_bigram(self, bigram: Bigram) -> dict:
        """
        Replace every instance of a given Bigram in the Vocabulary.

        For every Word which contains the target subword pair in its current subword mapping,
        replace the subword pair by concatenating its elements into one subword. Keep track
        of which other subword pairs are lost and gained in each Words's subword mapping and
        produce a dictionary of update dictionaries for each of those Bigrams.

        Args:
            bigram: The Bigram to be replaced

        Returns:
            A dictionary of update dictionaries for each Bigram with a change in frequency
        """

        a, b = bigram.pair
        tokens = bigram.token_freq.keys()
        updates = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            freq = self.tokn_dict[token].freq
            subwords = self.tokn_dict[token].subwords
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == a and subwords[i + 1] == b:
                    # Replace target bigram.
                    subwords[i] = a + b
                    del subwords[i + 1]
                    # Update other bigram frequencies.
                    if i > 0:
                        updates[(subwords[i - 1], a)][token] -= freq
                        updates[(subwords[i - 1], subwords[i])][token] += freq
                    if i < len(subwords) - 1:
                        updates[(b, subwords[i + 1])][token] -= freq
                        updates[(subwords[i], subwords[i + 1])][token] += freq
                i += 1
        return updates

    def map_to_sbwds(self, token: str) -> str:
        """
        Map a token string to a subwords string.

        Args:
            token: The token string to be mapped

        Returns:
            The subword string matching the token string
        """

        subwords = self.tokn_dict[token].subwords
        return ' '.join(subwords)

    def write(self, file: TextIO) -> None:
        """
        Write the Vocabulary to a file.

        Writes a vocabulary file with each line containing a unique string token followed
        by its frequency in the training set. For example, "THE 4827". The Vocabulary is
        sorted by frequency, from greatest to least, and then alphabetical order.

        Args:
            file: The file stream to be written to
        """

        for word in sorted(self.tokn_dict.values(), key=lambda x: (-x.freq, x.token)):
            file.write(f"{word.token} {word.freq}\n")


# Statistics object contains a dictionary of every subword pair currently appearing in the
# vocabulary and matches them to corresponding Bigram instances with up-to-date token frequency
# dictionaries. It also includes a search set containing a subset of bigrams with frequencies
# above the current frequency threshold and variables tracking the current threshold and an upper
# bound on bigram frequency. The previous search set's size and an adaptation parameter are used
# to adapt the next frequency threshold to produce a search set closer to the target size.
class Statistics:
    def __init__(self) -> None:
        """
        Initialize an empty Statistics instance.
        """

        self.bgrm_dict = {}
        self.max_freq = 0
        self.search_set = set()
        self.threshold = None
        self.adaptation_parameter = 0

    @classmethod
    def from_vocab(cls, vocab: Vocabulary) -> Statistics:
        """
        Initialize a Statistics instance with a given Vocabulary.

        Builds a bigram dictionary by counting the subword pairs found in each Word and scaling
        the counts by the Word's frequency. Before building the initial search set, it uses the
        most common Bigram's frequency to compute the search set's minimum frequency threshold.

        Args:
            vocab: Vocabulary to be processed

        Returns:
            A Statistics instance
        """

        new_stats = cls()
        for word in vocab.tokn_dict.values():
            token = word.token
            subwords = word.subwords
            freq = word.freq
            for i in range(len(subwords)-1):
                pair = (subwords[i], subwords[i + 1])
                if new_stats.missing(pair):
                    new_stats.add_bigram(pair)
                bigram = new_stats.bgrm_dict[pair]
                bigram.update_token_frequencies({token: freq})
                if bigram.freq > new_stats.max_freq:
                    new_stats.max_freq = bigram.freq
        new_stats.set_threshold(new_stats.max_freq)
        new_stats.build_search_set()
        return new_stats

    def missing(self, pair: tuple) -> bool:
        """
        Check if a subword pair is missing from the bigram dictionary.

        Args:
            pair: The subword pair to be searched for

        Returns:
            True if the subword pair is missing from the bigram dictionary and False otherwise
        """

        if pair in self.bgrm_dict:
            return False
        else:
            return True

    def add_bigram(self, pair: tuple) -> None:
        """
        Add a Bigram to the bigram dictionary.

        Args:
            pair: A subword pair representing the Bigram to be added
        """

        new_bigram = Bigram(pair)
        self.bgrm_dict[pair] = new_bigram

    def remove_bigram(self, bigram: Bigram) -> None:
        """
        Remove a Bigram from the bigram dictionary and the search set.

        Args:
            bigram: The Bigram to be removed
        """

        pair = bigram.pair
        del self.bgrm_dict[pair]
        bigram.remove_from_search_set(self.search_set)

    def update_pair_frequencies(self, pair_updates: dict) -> None:
        """
        Update the frequency statistics for each subword pair in the given update dictionary.

        For each subword pair in the update dictionary, update its corresponding Bigram instance.
        If a Bigram's frequency is greater than or equal to the search set's threshold, add it to
        the search set. Otherwise, if it is in the search set, remove it.

        Args:
            pair_updates: A dictionary of update dictionaries for each Bigram with a change in frequency
        """

        for pair, token_updates in pair_updates.items():
            if self.missing(pair):
                self.add_bigram(pair)
            bigram = self.bgrm_dict[pair]
            bigram.update_token_frequencies(token_updates)
            if bigram.freq >= self.threshold:
                if not bigram.in_search_set:
                    bigram.add_to_search_set(self.search_set)
            else:
                if bigram.in_search_set:
                    bigram.remove_from_search_set(self.search_set)
            # Remove a pair from the bigram dictionary if it no longer appears in any subword mappings.
            if bigram.freq == 0:
                del self.bgrm_dict[pair]

    def set_threshold(self, max_freq: Optional[int] = None) -> None:
        """
        Set a new frequency threshold for the search set.

        If a maximum bigram frequency is provided, use it in place of the previous threshold.

        Args:
            max_freq: The maximum bigram frequency
        """

        previous_threshold = self.threshold
        if max_freq:
            previous_threshold = max_freq
        reduction_factor = 1 + 2 ** self.adaptation_parameter
        self.threshold = min(ceil(previous_threshold / reduction_factor), previous_threshold - 1)

    def build_search_set(self) -> None:
        """
        Build a new search set.

        Iterate through the entire bigram dictionary and add to the search set each Bigram
        that meets the frequency threshold. Afterwards, adjust the adaptation parameter so
        that the next search set is closer to the target size.
        """

        for bigram in self.bgrm_dict.values():
            if bigram.freq >= self.threshold:
                bigram.add_to_search_set(self.search_set)
        search_set_size = len(self.search_set)
        if search_set_size < SEARCH_SET_TARGET_SIZE:
            self.adaptation_parameter += 1
        elif search_set_size > SEARCH_SET_TARGET_SIZE:
            self.adaptation_parameter -= 2

    def max_bigram(self) -> Bigram:
        """
        Find the most frequent Bigram.

        If the bigram dictionary is empty, all the words in the vocabulary have been
        concatenated into single subwords and the function will return None. Otherwise,
        the search set will be searched for the most frequent bigram. If the search set
        is empty, the frequency threshold will be reduced and a new search set will be
        built before running max_bigram() again.

        Returns:
            The most frequent Bigram
        """

        max_bigram = None
        if self.bgrm_dict:
            if self.search_set:
                max_bigram = next(iter(self.search_set))
                for bigram in self.search_set:
                    if bigram.freq > max_bigram.freq:
                        max_bigram = bigram
                    elif max_bigram.freq == self.max_freq:
                        break
            else:
                self.set_threshold()
                self.build_search_set()
                max_bigram = self.max_bigram()
            self.max_freq = max_bigram.freq
        return max_bigram


# Model object contains an ordered list of subword concatenation operations. Each operation
# is represented by a tuple of the two subword strings to be concatenated.
class BPEModel:
    def __init__(self) -> None:
        """
        Initialize an empty BPEModel instance.
        """

        self.operations = []

    @classmethod
    def from_model_file(cls, file: TextIO) -> BPEModel:
        """
        Initialize a BPEModel from a model file.

        Reads in a model file containing an ordered list of unique subword concatenation
        operations. Each line contains an operation represented by the two subword strings
        to be concatenated, for example "TH E_".

        Args:
            file: A file stream containing the model being used

        Returns:
            A BPEModel instance
        """

        new_model = cls()
        for line in file:
            line = line.upper()
            a, b = line.split()
            new_bigram = Bigram((a, b))
            new_model.add_operation(new_bigram)
        return new_model

    def add_operation(self, bigram: Bigram) -> None:
        """
        Add an operation to the BPEModel.

        Args:
            bigram: The Bigram to be added as a subword concatenation operation
        """

        self.operations.append(bigram.pair)

    def write(self, file: TextIO) -> None:
        """
        Write the BPEModel to a file.

        Writes a model file with an ordered list of unique subword concatenation
        operations. Each line contains an operation represented by the two subword
        strings to be concatenated, for example "TH E_".

        Args:
            file: The file stream to be written to
        """

        for operation in self.operations:
            file.write(f"{operation[0]} {operation[1]}\n")
