echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning WMT16 scripts...'
git clone https://github.com/rsennrich/wmt16-scripts.git

SCRIPTS=mosesdecoder/scripts
RO_SCRIPTS=wmt16-scripts
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl

NORMALIZE_IU_SPELLING=normalize-iu-spelling.pl

REMOVE_DIACRITICS=$RO_SCRIPTS/remove-diacritics.py
NORMALIZE_ROMANIAN=$RO_SCRIPTS/normalise-romanian.py

if [ ${lang} == "zh" ]; then         
    cat ${infile} \
    | ${REPLACE_UNICODE_PUNCT} \
    | ${NORM_PUNC} -l ${lang} \
    | ${REM_NON_PRINT_CHAR} \
    | hanzi-convert - -s \
    > ${outfile}
elif [ ${lang} == "ro" ]; then
    cat ${infile} \
    | ${REPLACE_UNICODE_PUNCT} \
    | ${NORM_PUNC} -l ${lang} \
    | ${REM_NON_PRINT_CHAR} \
    | ${NORMALIZE_ROMANIAN} \
    | ${REMOVE_DIACRITICS} \
    > ${outfile}
elif [ ${lang} == "iu" ]; then
    cat ${infile} \
    | ${REPLACE_UNICODE_PUNCT} \
    | ${NORM_PUNC} -l ${lang} \
    | ${REM_NON_PRINT_CHAR} \
    | perl ${NORMALIZE_IU_SPELLING} \
    > ${outfile}
else
    cat ${infile} \
    | ${REPLACE_UNICODE_PUNCT} \
    | ${NORM_PUNC} -l ${lang} \
    | ${REM_NON_PRINT_CHAR} \
    > ${outfile}
fi
