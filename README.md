# DP_BPE
A fast implementation of byte pair encoding (BPE)

Implements the algorithm described in:

[Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016.
Neural Machine Translation of Rare Words with Subword Units.
Proceedings the Association for Computational Linguistics.
](https://www.aclweb.org/anthology/P16-1162.pdf)

Documentation:

https://oalang.github.io/DP_BPE

## USAGE INSTRUCTIONS

The code can be run in a Python console using the methods provided by the `bpe` package
or in a terminal using the executable scripts.

### Python Console Example

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

### Running the Scripts in a Terminal

To compile a vocabulary from a text file:

    compile_vocabulary.py --text <text_file> --output <vocabulary_file>

The vocabulary file will look something like this:

    THE 60133
    AND 33258
    OF 29876
    TO 27365
    A 21766
    ...

To train a bpe model:

    bpe_train_model.py --vocabulary <vocabulary_file> --max-subwords <number> --output <bpe_model_file>

The model file will look something like this:

    E _
    T H
    D _
    S _
    TH E_  
    ...

To encode a text file:

    python bpe_encode_text.py --bpe-model <bpe_model_file> --text <text_file> --output <subword_file>

The encoded file will look something like this:

    CH A P T ER_ ON E_ MISSUS_ RACHEL_ LY ND E_ I S_ SU R P R IS ED_
    THAT_ HAD_ I T S_ S OU R C E_ A WA Y_ B A C K_ IN_ THE_ W O O D S_
    FOR _ N O T_ E V E N_ A_ B RO O K_ C OU L D_ R U N_ P A S T_
    AND_ THAT_ I F_ SHE_ N O T I C ED_ AN Y TH ING_ O D D_ OR _ OU T_
    B U T_ MISSUS_ RACHEL_ LY ND E_ WAS_ ON E_ OF_ TH O S E_

To decode a subwords file:

    python bpe_decode_subwords.py --subwords <subword_file> --output <text_file>
