from collections import defaultdict

import torch
from fairseq.logging import metrics
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from ..datasets import MultilingualDatasetManager


@register_task("mask_translation_multi_simple_epoch")
class MaskTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )

    def valid_step(self, sample, model, criterion):
        with torch.no_grad():
            model.eval()
            model.patch_all_mask(src_lang=sample['src_lang'], tgt_lang=sample['tgt_lang'])
            return super().valid_step(sample, model, criterion)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        src_lang = sample['src_lang']
        tgt_lang = sample['tgt_lang']
        model.train()
        model.set_num_updates(update_num)
        model.patch_all_mask(src_lang=src_lang, tgt_lang=tgt_lang)
        res = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        return res

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        for m in models:
            m.eval()
        models = [m.patch_all_mask(
            src_lang=self.args.source_lang,
            tgt_lang=self.args.target_lang
        )[0] for m in models]
        return super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs)
