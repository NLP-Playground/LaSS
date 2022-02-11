# learn bpe

python -u fairseq/scripts/spm_train.py --input=/path/to/your/file --model_prefix=spm.bpe --vocab_size=64000 --character_coverage=1.0 --model_type=bpe --num_threads=45 --shuffle_input_sentence --train_extremely_large_corpus

# apply bpe
python fairseq/scripts/spm_encode.py --model spm.bpe.model \
    --output_format=piece --inputs en-${lang}/clean/train/train.full.${lang} en-${lang}/clean/train/train.full.en \
    --outputs ${train_path}/train.full.bpe.${lang} ${train_path}/train.full.bpe.en \
    --min-len 1 --max-len 256

# restrict length ratio
SCRIPTS=mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
perl $CLEAN -ratio 3.0 ${prefix}/norm.dedup.spm en ${lang} ${prefix}/norm.dedep.spm.clean 1 256