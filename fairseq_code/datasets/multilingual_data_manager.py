from collections import OrderedDict

import cytoolz as toolz
import more_itertools

from fairseq.data import SampledMultiEpochDataset
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager as OldMultilingualDatasetManager
from fairseq.data.multilingual.sampled_multi_dataset import CollateFormat
from fairseq.data.multilingual.sampled_multi_dataset import SampledMultiDataset as OldSampledMultiDataset

from .batched_sampled_multi_epoch_dataset import BatchedSampledMultiEpochDataset


class SampledMultiDataset(OldSampledMultiDataset):

    def _group_indices_by_dataset_index(self, indices):
        return toolz.groupby(lambda x: self._get_dataset_and_index(x)[0], indices)

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        batches_list = []
        self.batched_size_dict = {}
        for k, grouped_indices in self._group_indices_by_dataset_index(indices).items():
            # Group indices by the dataset.
            batches = super().batch_by_size(grouped_indices, max_tokens, max_sentences, required_batch_size_multiple)
            batches_list.append(
                batches
            )
            self.batched_size_dict[k] = len(batches)
        return list(more_itertools.flatten(batches_list))

    def collater(self, samples, **extra_args):
        if len(samples) == 0:
            return None
        # Add language to the batch
        batch = super().collater(samples, **extra_args)
        assert len(batch) == 1
        key = list(batch.keys())[0]
        # The format of key {data_category}:{src}-{tgt}
        src_lang, tgt_lang = key.split(":")[1].strip().split("-")
        batch = batch[key]
        batch["src_lang"] = src_lang
        batch["tgt_lang"] = tgt_lang
        return batch


class MultilingualDatasetManager(OldMultilingualDatasetManager):
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return cls(args, lang_pairs, langs, dicts, sampling_method)

    def load_into_concat_dataset(self, split, datasets, data_param_list):
        return SampledMultiDataset(
            OrderedDict(datasets),
            sampling_ratios=None,
            eval_key=None,
            collate_format="ordered_dict",
            virtual_size=None,
            split=split,
        )

    def load_sampled_multi_epoch_dataset(self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs):
        # Datasets is a list of tuple with type Tuple[str, FairseqDataset], the string is
        # a key in data_param_list attribute.
        # Data_param_list is a list of dict, the dict contains {"key" ,...}
        datasets, data_param_list = self.load_split_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )
        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            return BatchedSampledMultiEpochDataset(
                OrderedDict(datasets),
                epoch=epoch,
                shard_epoch=shard_epoch,
                # valid and test datasets will be degenerate to concating datasets:
                sampling_ratios=sample_ratios,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=self.args.virtual_data_size,
                split=split,
                virtual_epoch_size=self.args.virtual_epoch_size,
                # if not using lang_tok altering, simplified to use the same collater
                shared_collater=self._shared_collater(),
            )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)

