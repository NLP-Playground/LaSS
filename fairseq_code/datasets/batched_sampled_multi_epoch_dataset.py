from fairseq.data import SampledMultiEpochDataset
import cytoolz as toolz
import more_itertools


class BatchedSampledMultiEpochDataset(SampledMultiEpochDataset):
    """
    The only difference compared with SampledMultiEpochDataset is
    batch size. This dataset will only group data from one dataset
    to one batch.
    """

    def _group_indices_by_dataset_index(self, indices):
        return toolz.groupby(lambda x: self._get_dataset_and_index(x)[0], indices)

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        batches = []
        for _, grouped_indices in self._group_indices_by_dataset_index(indices).items():
            # Group indices by the dataset.
            batches.append(
                super().batch_by_size(grouped_indices, max_tokens, max_sentences, required_batch_size_multiple)
            )
        return list(more_itertools.flatten(batches))

    def collater(self, samples, **extra_args):
        if len(samples) == 0:
            return None
        # Add language to the batch
        batch = super().collater(samples, **extra_args)
        assert len(set(sample[0] for sample in samples)) == 1
        key = self.keys[samples[0][0]]
        # The format of key {data_category}:{src}-{tgt}
        src_lang, tgt_lang = key.split(":")[1].strip().split("-")
        batch["src_lang"] = src_lang
        batch["tgt_lang"] = tgt_lang
        return batch
