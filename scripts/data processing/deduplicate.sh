paste ${prefix}/normalize/train/en.norm ${prefix}/normalize/train/${lang}.norm | awk '!x[$0]++' > ${prefix}/deduplicate/train/train.norm.dedup
echo "keeping $(wc -l ${prefix}/deduplicate/train/train.norm.dedup) bitext out of $(wc -l ${prefix}/normalize/train/en.norm)"
cut -f1 ${prefix}/deduplicate/train/train.norm.dedup > ${prefix}/deduplicate/train/en.norm.dedup
cut -f2 ${prefix}/deduplicate/train/train.norm.dedup > ${prefix}/deduplicate/train/${lang}.norm.dedup
rm ${prefix}/deduplicate/train/train.norm.dedup