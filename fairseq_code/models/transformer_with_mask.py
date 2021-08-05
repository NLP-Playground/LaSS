import json
import logging
from collections import defaultdict

import typing

from fairseq.file_io import PathManager
from fairseq.models import register_model, register_model_architecture

import torch.nn as nn
import torch

import cytoolz as toolz

from .transformer_mask_base_model import TransformerMaskBaseModel

logger = logging.getLogger(__name__)


@register_model("transformer_with_mask")
class TransformerWithMaskModel(TransformerMaskBaseModel):
    """
    This is a transformer model weight mask.
    The mask is described with a dict, the key is like the key in the state_dict.
    The mask will mask the weight in the state_dict with the same key.
    """

    @staticmethod
    def add_args(parser):
        super(TransformerWithMaskModel, TransformerWithMaskModel).add_args(parser)
        parser.add_argument("--mask-path", help="A json dict of the path to all language direction")
        parser.add_argument("--no-save-static-mask", action="store_true")
        parser.add_argument("--mask-embedding", action="store_true",
                            help="Mask the embedding module")
        parser.add_argument("--no-mask-output-project", action="store_true", help="No mask the output project")

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        mask_dict = getattr(args, "mask_path", None)
        if mask_dict is not None:
            mask_dict = json.loads(mask_dict)
            for k, mask_path in mask_dict.items():
                with PathManager.open(mask_path, "rb") as f:
                    mask = torch.load(
                        f,
                        map_location=(
                            lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                        ),
                    )
                    model.add_language_mask(k, mask)
            # model.language_mask.to(model.device)

        return model

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.no_save_static_mask = getattr(args, "no_save_static_mask", False)
        self.language_mask = nn.ModuleDict({})

    def add_language_mask(self, k, v):
        if k in self.language_mask:
            return
        self.language_mask[k] = nn.ParameterDict({self._format_name(a): nn.Parameter(b, requires_grad=False)
                                                  for a, b in v.items()})

    def upgrade_state_dict_named(self, state_dict, name):
        # expand the self.language_mask to match the language mask in state_dict

        loaded_mask_name = []
        for k in state_dict.keys():
            if k.startswith("language_mask"):
                loaded_mask_name.append(k)
        grouped_loaded_mask_name = toolz.groupby(lambda x: x.strip().split(".")[1], loaded_mask_name)
        for language, keys in grouped_loaded_mask_name.items():
            # self.language_mask[language] = nn.ParameterDict({k.strip().split(".")[2]: state_dict[k] for k in keys})
            self.add_language_mask(language, {k.strip().split(".")[2]: state_dict[k] for k in keys})
        # self.language_mask.to(self.device)

        # expand the self.language_mask to match the language mask in state_dict
        # Also need to add the language_mask to the state_dict if not in state_dict
        for k, v in self.language_mask.items():
            for i_k, d in v.items():
                key = f"language_mask.{k}.{i_k}"
                if key in self.language_mask:
                    pass
                else:
                    state_dict[key] = d

        # move global weight to weight, all weight will be moved to global weight when it need used
        for k in filter(lambda x: "global_weight" in x, list(state_dict.keys())):
            new_k = k.replace("global_weight", "weight")
            state_dict[new_k] = state_dict[k]
            del state_dict[k]

        super().upgrade_state_dict_named(state_dict, name)

    def patch_all_mask(self, src_lang: str, tgt_lang: str):
        threshold_percent = self._patch_static_mask(src_lang, tgt_lang)
        return self, threshold_percent

    def _patch_static_mask(self, src_lang, tgt_lang):
        lang_pair = self.lang_pair(src=src_lang, tgt=tgt_lang)
        assert lang_pair in self.language_mask

        total_number = 0
        un_mask_number = 0

        for name, c in self.named_modules():
            if isinstance(c, nn.Linear) or (self.args.mask_embedding and isinstance(c, nn.Embedding)):
                if getattr(self.args, "no_mask_output_project", False) and "output_projection" == name:
                    continue
                if not hasattr(c, "global_weight"):
                    c.global_weight = c.weight
                    del c.weight
                mask_key = name + ".weight"
                mask_key = self._format_name(mask_key)
                if mask_key in self.language_mask[lang_pair]:
                    mask = self.language_mask[lang_pair][mask_key]
                    total_number += mask.numel()
                    un_mask_number += mask.sum().item()
                    c.weight = mask * c.global_weight
                    # def repopulate_weight(mod, _):
                    #     nonlocal mask
                    #     mod.weight = mask * mod.global_weight
                    # c.register_forward_pre_hook(repopulate_weight)
                else:
                    c.weight = c.global_weight

        return un_mask_number / total_number

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)

        # Remove the static dict from checkpoint
        if self.no_save_static_mask:
            for k in list(state_dict.keys()):
                if k.startswith("language_mask"):
                    del state_dict[k]
        return state_dict


@register_model_architecture("transformer_with_mask", "transformer_with_mask")
def base_architecture(args):
    from fairseq.models.transformer import base_architecture as transformer_base_architecture
    transformer_base_architecture(args)
    args.mask_path = getattr(args, "mask_path", None)
    args.mask_embedding = getattr(args, "mask_embedding", False)
    args.no_mask_output_project = getattr(args, "no_mask_output_project", False)


@register_model_architecture("transformer_with_mask", "transformer_iwslt_arch_with_mask")
def transformer_iwslt_arch_with_mask(args):
    from fairseq.models.transformer import transformer_iwslt_de_en
    transformer_iwslt_de_en(args)
    base_architecture(args)


@register_model_architecture("transformer_with_mask", "transformer_vaswani_wmt_en_fr_big_with_mask")
def transformer_vaswani_wmt_en_fr_big_with_mask(args):
    from fairseq.models.transformer import transformer_vaswani_wmt_en_fr_big
    transformer_vaswani_wmt_en_fr_big(args)
    base_architecture(args)


@register_model_architecture("transformer_with_mask", "transformer_wmt_en_de_big_t2t_with_mask")
def transformer_wmt_en_de_big_t2t_with_mask(args):
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)
    base_architecture(args)


@register_model_architecture("transformer_with_mask", "architecture_for_test")
def architecture_for_test(args):
    args.encoder_embed_dim = 256
    args.encoder_ffn_embed_dim = 1024
    args.decoder_ffn_embed_dim = 1024
    args.encoder_layers = 3
    args.decoder_layers = 3
    args.encoder_attention_heads = 4
    args.decoder_attention_heads = 4
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.share_all_embeddings = True

    args.soft_threshold = True
    args.soft_threshold_level = "vector_shared_g"
    args.soft_threshold_init_bias = -12800

    # args.mask_path = """{
    # "ar-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "de-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-ar":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-de":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-es":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-fa":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-he":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-it":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-nl":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "en-pl":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "es-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "fa-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "he-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "it-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "nl-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt",
    # "pl-en":"/data00/home/wuliwei.000/test_data_sparse_sharing/mask.pt"
    # }"""
    base_architecture(args)
