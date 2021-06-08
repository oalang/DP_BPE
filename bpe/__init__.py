"""
A fast implementation of byte pair encoding (BPE).

https://github.com/oalang/DP_BPE

Implements the algorithm described in:
    | Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016.
    | Neural Machine Translation of Rare Words with Subword Units.
    | Proceedings the Association for Computational Linguistics.

Example:
    >>> from bpe import *
    >>> text_file = open('sample_text.txt')
    >>> vocabulary = Vocabulary.from_text_file(text_file)
    >>> bpe_model = train_model(vocabulary, 100)
    >>> subwords = encode_text('Hello, world.', bpe_model)
    >>> subwords
    'H E L L O_ W OR L D_'
    >>> text = decode_subwords(subwords)
    >>> text
    'HELLO WORLD'

Example:
    compile_vocabulary.py --text sample_text.txt --output vocabulary.txt
    bpe_train_model.py --vocabulary vocabulary.txt --max-subwords 100 --output bpe_model.txt
    bpe_encode_text.py --bpe-model bpe_model.txt --text sample_text.txt --output subwords.txt
    bpe_decode_subwords.py --subwords subwords.txt --output text.txt
"""

from __future__ import annotations
from typing import Dict, Optional, Set, TextIO, Tuple
from collections import defaultdict
from re import sub
from math import ceil

StringPair = Tuple[str, str]

_SEARCH_SET_TARGET_SIZE = 100
"""int: This search set size was found to produce speedy results."""


class Word:
    """Represents a word in a Vocabulary.

    Attributes:
        token (str): A unique token string used to identify this Word.
        frequency (int): The Word's frequency in the text used to generate the Vocabulary.
        subwords (str): The Word's current subword mapping.
    """

    def __init__(self, token: str) -> None:
        """Initializes a Word instance.

        The new instance will have an initial frequency of zero and a subword string
        split into individual characters followed by the end-of-token symbol '_'.

        Args:
            token: A unique token string used to identify this Word.
        """

        self.token = token
        self.frequency = 0
        self.subwords = list(token) + ['_']

    def update_frequency(self, n: int) -> None:
        """Adds n to the Word's frequency.

        Args:
            n: The number to add to the Word's frequency.
        """

        self.frequency += n

    def apply_model(self, bpe_model: BPEModel) -> None:
        """Applies a BPEModel to the Word to arrange its characters into subwords.

        Args:
            bpe_model: The BPEModel to be used.
        """

        # Run through each subword concatenation operation in order and apply it to
        # the subword mapping.
        subwords = self.subwords
        for subword_pair in bpe_model.operations:
            subword_a, subword_b = subword_pair
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == subword_a and subwords[i + 1] == subword_b:
                    # Replace target bigram.
                    subwords[i] = subword_a + subword_b
                    del subwords[i + 1]
                i += 1


class Bigram:
    """Represents a bigram in a Statistics.

    Keeping track of which tokens contain a bigram results in a significant time
    reduction when removing a bigram from the vocabulary, compared to processing
    every token in the vocabulary.

    Attributes:
        subword_pair (StringPair): A pair of subword strings representing the Bigram.
        frequency (int): The Bigram's frequency in the text used to generate the Vocabulary.
        token_frequency (Dict[str, int]): A dictionary of tokens which currently contain the
            Bigram in their subword mappings.
        in_search_set (bool): A boolean indicating whether or not the Bigram is in the
            current search set for most frequent Bigram.
    """

    def __init__(self, subword_pair: StringPair) -> None:
        """Initializes a Bigram instance.

        The new instance will have an initial frequency of zero.

        Args:
            subword_pair: A pair of subword strings.
        """

        self.subword_pair = subword_pair
        self.frequency = 0
        self.token_frequency = defaultdict(int)
        self.in_search_set = False

    def update_token_frequencies(self, token_updates: Dict[str, int]) -> None:
        """Updates the Bigram's frequency statistics.

        Updates the frequency of the Bigram for every Word where it has changed, and
        also its overall frequency.

        Args:
            token_updates: A dictionary whose keys are tokens whose Word contains the
            subword pair in their subword mapping and whose values are the updates.
        """

        for token, n in token_updates.items():
            self.token_frequency[token] += n
            # Remove a token from the frequency dictionary if it no longer contains the bigram.
            if self.token_frequency[token] == 0:
                del self.token_frequency[token]
            self.frequency += n

    def add_to_search_set(self, search_set: Set[Bigram]) -> None:
        """Adds the Bigram to the search set.

        Adds the Bigram to the search set and sets its in_search_set flag to True.

        Args:
            search_set: The search set.
        """

        search_set.add(self)
        self.in_search_set = True

    def remove_from_search_set(self, search_set: Set[Bigram]) -> None:
        """Removes the Bigram from the search set.

        Removes the Bigram from the search set and sets its in_search_set flag to False.

        Args:
            search_set: The search set.
        """

        search_set.remove(self)
        self.in_search_set = False


class Vocabulary:
    """A data structure used to keep track of the unique token strings found in a given
    text and their frequencies.

    Attributes:
        words (Dict[str, Word]): A dictionary of token strings found in the given text
            and their corresponding Word instances.
        characters (Set[str]): A set of all the characters found in the given text.
    """

    def __init__(self) -> None:
        """Initializes an empty Vocabulary instance."""

        self.words = {}
        self.characters = set()

    @classmethod
    def from_text_file(cls, file: TextIO) -> Vocabulary:
        """Initializes a Vocabulary from a text file.

        Generates a new Word instance for each unique token string and counts its
        occurrences.

        Args:
            file: A file stream containing the text being processed.

        Returns:
            A Vocabulary instance.
        """

        new_vocabulary = cls()
        for line in file:
            new_vocabulary.add_text(line)
        return new_vocabulary

    @classmethod
    def from_vocabulary_file(cls, file: TextIO) -> Vocabulary:
        """Initializes a Vocabulary from a vocabulary file.

        Reads in a vocabulary file with each line containing a unique token string followed
        by its frequency in the training set. For example, "THE 4827".

        Args:
            file: A file stream containing the vocabulary being processed.

        Returns:
            A Vocabulary instance.
        """

        new_vocabulary = cls()
        for line in file:
            line = line.upper()
            token, frequency = line.split()
            frequency = int(frequency)
            new_vocabulary.add_word(token)
            new_vocabulary.words[token].update_frequency(frequency)
        return new_vocabulary

    def add_text(self, text: str) -> None:
        """Adds the contents of a text string to the Word frequencies.

        Normalizes the text by capitalizing everything and removing punctuation.

        Args:
            text: The text string to be added.
        """

        text = text.upper()
        text = sub(r"[^A-Z']", " ", text)
        for token in text.split():
            if self.missing(token):
                self.add_word(token)
            self.words[token].update_frequency(1)

    def reset_subwords(self) -> None:
        """Resets every Word's subword mapping to individual characters."""

        for word in self.words.values():
            word.subwords = list(word.token) + ['_']

    def missing(self, token: str) -> bool:
        """Checks if a token string is missing from the token dictionary.

        Args:
            token: The token string to be searched for.

        Returns:
            True if the token string is missing from the token dictionary and False otherwise.
        """

        if token in self.words:
            return False
        else:
            return True

    def add_word(self, token: str, bpe_model: Optional[BPEModel] = None) -> None:
        """Adds a Word to the token dictionary.

        If a BPEModel is provided, it is applied to the Word to produce a subword mapping.

        Args:
            token: A token string representing the Word to be added.
            bpe_model: Optional; A BPEModel used produce a subword mapping.
        """

        new_word = Word(token)
        self.words[token] = new_word
        self.characters.update(new_word.subwords)
        if bpe_model:
            new_word.apply_model(bpe_model)

    def num_characters(self) -> int:
        """Returns the size of the Vocabulary's character set.

        Returns:
            The size of the Vocabulary's character set.
        """

        return len(self.characters)

    def replace_bigram(self, bigram: Bigram) -> Dict[StringPair, Dict[str, int]]:
        """Replaces every instance of a given Bigram in the Vocabulary.

        Args:
            bigram: The Bigram to be replaced.

        Returns:
            A dictionary of token update dictionaries for each Bigram with a change in frequency.
        """

        # For every Word which contains the target subword pair in its current subword mapping,
        # replace the subword pair by concatenating its elements into one subword. Keep track
        # of which other subword pairs are lost and gained in each Words's subword mapping and
        # produce a dictionary of update dictionaries for each of those Bigrams.
        subword_a, subword_b = bigram.subword_pair
        tokens = bigram.token_frequency.keys()
        bigram_updates = defaultdict(lambda: defaultdict(int))
        for token in tokens:
            frequency = self.words[token].frequency
            subwords = self.words[token].subwords
            i = 0
            while i < len(subwords) - 1:
                if subwords[i] == subword_a and subwords[i + 1] == subword_b:
                    # Replace target bigram.
                    subwords[i] = subword_a + subword_b
                    del subwords[i + 1]
                    # Update other bigram frequencies.
                    if i > 0:
                        bigram_updates[(subwords[i - 1], subword_a)][token] -= frequency
                        bigram_updates[(subwords[i - 1], subwords[i])][token] += frequency
                    if i < len(subwords) - 1:
                        bigram_updates[(subword_b, subwords[i + 1])][token] -= frequency
                        bigram_updates[(subwords[i], subwords[i + 1])][token] += frequency
                i += 1
        return bigram_updates

    def map_to_subwords(self, token: str) -> str:
        """Maps a token string to a subwords string.

        Args:
            token: The token string to be mapped.

        Returns:
            The subword string matching the token string.
        """

        subwords = self.words[token].subwords
        return ' '.join(subwords)

    def write(self, file: TextIO) -> None:
        """Writes the Vocabulary to a file.

        Writes a vocabulary file with each line containing a unique token string followed
        by its frequency in the training set. For example, "THE 4827". The Vocabulary is
        sorted by frequency, from greatest to least, and then alphabetical order.

        Args:
            file: The file stream to be written to.
        """

        for word in sorted(self.words.values(), key=lambda x: (-x.frequency, x.token)):
            file.write(f"{word.token} {word.frequency}\n")


class Statistics:
    """A data structure used to keep track of Bigram frequencies in a Vocabulary.

    Statistics keeps track of the changes in Bigram frequency as a Vocabulary is
    altered by subword concatenation operations and provides a fast method for
    finding the most frequent bigram.

    Attributes:
        bigrams (Dict[StringPair, Bigram]): A dictionary of subword pairs currently
            found in the Vocabulary and their corresponding Bigram instances.
        max_frequency (int): The frequency of the most common Bigram.
        search_set (Set[Bigram]): A subset of bigrams guaranteed to contain the most
            frequent Bigram.
        threshold (int): The minimum frequency threshold for inclusion in search_set.
        adaptation_parameter (int): A parameter used to compute the next threshold
            which is adaptively tuned to make the the next search_set closer to the
            target size.
    """
    def __init__(self) -> None:
        """Initializes an empty Statistics instance."""

        self.bigrams = {}
        self.max_frequency = 0
        self.search_set = set()
        self.threshold = None
        self.adaptation_parameter = 0

    @classmethod
    def from_vocabulary(cls, vocabulary: Vocabulary) -> Statistics:
        """Initializes a Statistics instance with a given Vocabulary.

        Args:
            vocabulary: Vocabulary to be processed.

        Returns:
            A Statistics instance.
        """

        # Build a bigram dictionary by counting the subword pairs found in each Word and scaling
        # the counts by the Word's frequency. Before building the initial search set, use the
        # most common Bigram's frequency to compute the search set's minimum frequency threshold.
        new_statistics = cls()
        for word in vocabulary.words.values():
            token = word.token
            subwords = word.subwords
            frequency = word.frequency
            for i in range(len(subwords)-1):
                pair = (subwords[i], subwords[i + 1])
                if new_statistics.missing(pair):
                    new_statistics.add_bigram(pair)
                bigram = new_statistics.bigrams[pair]
                bigram.update_token_frequencies({token: frequency})
                if bigram.frequency > new_statistics.max_frequency:
                    new_statistics.max_frequency = bigram.frequency
        new_statistics.set_threshold(new_statistics.max_frequency)
        new_statistics.build_search_set()
        return new_statistics

    def missing(self, subword_pair: StringPair) -> bool:
        """Checks if a subword pair is missing from the bigram dictionary.

        Args:
            subword_pair: The subword pair to be searched for.

        Returns:
            True if the subword pair is missing from the bigram dictionary and False otherwise.
        """

        if subword_pair in self.bigrams:
            return False
        else:
            return True

    def add_bigram(self, subword_pair: StringPair) -> None:
        """Adds a Bigram to the bigram dictionary.

        Args:
            subword_pair: A subword pair representing the Bigram to be added.
        """

        new_bigram = Bigram(subword_pair)
        self.bigrams[subword_pair] = new_bigram

    def remove_bigram(self, bigram: Bigram) -> None:
        """Removes a Bigram from the bigram dictionary and the search set.

        Args:
            bigram: The Bigram to be removed.
        """

        subword_pair = bigram.subword_pair
        del self.bigrams[subword_pair]
        bigram.remove_from_search_set(self.search_set)

    def update_bigram_frequencies(self, bigram_updates: dict) -> None:
        """Updates the frequency statistics for each subword pair in the given update dictionary.

        Args:
            bigram_updates: A dictionary of token update dictionaries for each Bigram with a
                change in frequency.
        """

        # For each subword pair in the update dictionary, update its corresponding Bigram instance.
        # If a Bigram's frequency is greater than or equal to the search set's threshold, add it to
        # the search set. Otherwise, if it is in the search set, remove it.
        for subword_pair, token_updates in bigram_updates.items():
            if self.missing(subword_pair):
                self.add_bigram(subword_pair)
            bigram = self.bigrams[subword_pair]
            bigram.update_token_frequencies(token_updates)
            if bigram.frequency >= self.threshold:
                if not bigram.in_search_set:
                    bigram.add_to_search_set(self.search_set)
            else:
                if bigram.in_search_set:
                    bigram.remove_from_search_set(self.search_set)
            # Remove a pair from the bigram dictionary if it no longer appears in any subword mappings.
            if bigram.frequency == 0:
                del self.bigrams[subword_pair]

    def set_threshold(self, max_frequency: Optional[int] = None) -> None:
        """Sets a new frequency threshold for the search set.

        If a maximum bigram frequency is provided, use it in place of the previous threshold.

        Args:
            max_frequency: Optional; The maximum bigram frequency.
        """

        previous_threshold = self.threshold
        if max_frequency:
            previous_threshold = max_frequency
        reduction_factor = 1 + 2 ** self.adaptation_parameter
        self.threshold = min(ceil(previous_threshold / reduction_factor), previous_threshold - 1)

    def build_search_set(self) -> None:
        """Builds a new search set."""

        # Iterate through the entire bigram dictionary and add to the search set each Bigram
        # that meets the frequency threshold. Afterwards, adjust the adaptation parameter so
        # that the next search set is closer to the target size.
        for bigram in self.bigrams.values():
            if bigram.frequency >= self.threshold:
                bigram.add_to_search_set(self.search_set)
        search_set_size = len(self.search_set)
        if search_set_size < _SEARCH_SET_TARGET_SIZE:
            self.adaptation_parameter += 1
        elif search_set_size > _SEARCH_SET_TARGET_SIZE:
            self.adaptation_parameter -= 2

    def max_bigram(self) -> Bigram:
        """Finds the most frequent Bigram.

        Returns:
            The most frequent Bigram.
        """

        # If the bigram dictionary is empty, all the words in the vocabulary have been
        # concatenated into single subwords; return None. Otherwise, search the search
        # set for the most frequent bigram. If the search set is empty, reduce the
        # frequency threshold, rebuild the search set, and run max_bigram() again.
        max_bigram = None
        if self.bigrams:
            if self.search_set:
                max_bigram = next(iter(self.search_set))
                for bigram in self.search_set:
                    if bigram.frequency > max_bigram.frequency:
                        max_bigram = bigram
                    elif max_bigram.frequency == self.max_frequency:
                        break
            else:
                self.set_threshold()
                self.build_search_set()
                max_bigram = self.max_bigram()
            self.max_frequency = max_bigram.frequency
        return max_bigram


class BPEModel:
    """A Bight Pair Encoding model.

    The model consists of an ordered list of subword concatenation operations. Each operation
    is represented by a tuple of the two subword strings to be concatenated.

    Attributes:
        operations (List[StringPair]): An ordered list of subword concatenation operations.
    """

    def __init__(self) -> None:
        """Initializes an empty BPEModel instance."""

        self.operations = []

    @classmethod
    def from_model_file(cls, file: TextIO) -> BPEModel:
        """Initializes a BPEModel from a model file.

        Reads in a model file containing an ordered list of unique subword concatenation
        operations. Each line contains an operation represented by the two subword strings
        to be concatenated, for example "TH E_".

        Args:
            file: A file stream containing the model being used.

        Returns:
            A BPEModel instance.
        """

        new_bpe_model = cls()
        for line in file:
            line = line.upper()
            subword_a, subword_b = line.split()
            new_bigram = Bigram((subword_a, subword_b))
            new_bpe_model.add_operation(new_bigram)
        return new_bpe_model

    def add_operation(self, bigram: Bigram) -> None:
        """Adds an operation to the BPEModel.

        Args:
            bigram: The Bigram to be added as a subword concatenation operation.
        """

        self.operations.append(bigram.subword_pair)

    def write(self, file: TextIO) -> None:
        """Writes the BPEModel to a file.

        Writes a model file with an ordered list of unique subword concatenation
        operations. Each line contains an operation represented by the two subword
        strings to be concatenated, for example "TH E_".

        Args:
            file: The file stream to be written to.
        """

        for operation in self.operations:
            file.write(f"{operation[0]} {operation[1]}\n")


def train_model(vocabulary: Vocabulary, max_subwords: int = 1000) -> BPEModel:
    """Trains a BPEModel on a Vocabulary.

    The Vocabulary object's subword mappings will be altered to match the model, so make a
    copy to input if you want the original to remain unchanged.

    Args:
        vocabulary: The Vocabulary to be trained on.
        max_subwords: The maximum number of unique subwords that may appear in an encoding
            using this BPEModel. Default = 1000.

    Returns:
        A BPEModel.
    """

    # Train the BPEModel by iteratively adding the most frequent Bigram to the model, replacing
    # that Bigram in the Vocabulary's subword mappings with its concatenation, removing it from
    # the Statistics, and then updating the frequencies of every other Bigram affected by the
    # operation.
    vocabulary.reset_subwords()
    statistics = Statistics.from_vocabulary(vocabulary)
    bpe_model = BPEModel()
    max_operations = max_subwords - vocabulary.num_characters()
    for i in range(max_operations):
        max_bigram = statistics.max_bigram()
        if max_bigram is None:
            print(f"Stopped early with {i} operations")
            break
        bpe_model.add_operation(max_bigram)
        bigram_updates = vocabulary.replace_bigram(max_bigram)
        statistics.remove_bigram(max_bigram)
        statistics.update_bigram_frequencies(bigram_updates)
    return bpe_model


def encode_text(text: str, bpe_model: BPEModel, vocabulary: Optional[Vocabulary] = None) -> str:
    """Encodes a given text string into subwords using a given BPEModel.

    Normalizes the text by capitalizing everything and removing punctuation.

    If providing a Vocabulary, its subwords mappings should have been generated using the
    given BPEModel. New Word instances may be added to the Vocabulary, so make a copy to
    input if you want the original to remain unchanged.

    Args:
        text: The text string to be encoded into subwords.
        bpe_model: The BPEModel used for the encoding.
        vocabulary: Optional; A Vocabulary containing Word instances with subword mappings
            already generated by the BPEModel.

    Returns:
        The encoded text string.
    """

    if vocabulary is None:
        vocabulary = Vocabulary()
    encodings = []
    text = text.upper()
    text = sub(r"[^A-Z']", " ", text)
    for token in text.split():
        if vocabulary.missing(token):
            vocabulary.add_word(token, bpe_model)
        encodings.append(vocabulary.map_to_subwords(token))
    return ' '.join(encodings)


def decode_subwords(subwords: str) -> str:
    """Decodes a given subword string into regular text.

    Args:
        subwords: The subword string to be decoded into regular text.

    Returns:
        The decoded text string.
    """

    return subwords.replace(" ", "").replace("_", " ").strip()
