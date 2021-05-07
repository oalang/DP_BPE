# DP_BPE
A fast implementation of byte pair encoding (BPE)

USAGE INSTRUCTIONS
------------------

To compile a vocabulary from a text file:

    python compile_vocab.py --text <text_file> --output <vocab_file>

The vocabulary file will look like this:

    THE 60133
    AND 33258
    OF 29876
    TO 27365
    A 21766
    ...

To train a bpe model:

    python train_model.py --vocab <vocab_file> --max-subwords <number> --output <model_file>

The model file will look like this:

    E _
    T H
    D _
    S _
    TH E_  
    ...

To encode a text file:

    python encode_text.py --model <model_file> --text <text_file> --output <subwrd_file>

The encoded file will look like this:

    CH A P T ER_ ON E_ MISSUS_ RACHEL_ LY ND E_ I S_ SU R P R IS ED_
    THAT_ HAD_ I T S_ S OU R C E_ A WA Y_ B A C K_ IN_ THE_ W O O D S_
    FOR _ N O T_ E V E N_ A_ B RO O K_ C OU L D_ R U N_ P A S T_
    AND_ THAT_ I F_ SHE_ N O T I C ED_ AN Y TH ING_ O D D_ OR _ OU T_
    B U T_ MISSUS_ RACHEL_ LY ND E_ WAS_ ON E_ OF_ TH O S E_

To decode a subwords file:

    python decode_subwords.py --subwords <subwrd_file> --output <text_file>
