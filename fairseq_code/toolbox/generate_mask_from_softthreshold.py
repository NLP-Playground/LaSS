#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.data import encoders
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from torch import nn

def loop_linear_module_for_model(m, no_mask_output_project=False, with_name=False, prefix=""):
    for n, c in m.named_children():

        if no_mask_output_project and n == "output_projection":
            continue
        now_name = n if len(prefix) == 0 else prefix + "." + n

        if isinstance(c, nn.Linear):
            if with_name:
                yield now_name, c
            else:
                yield c
        yield from loop_linear_module_for_model(c, no_mask_output_project=no_mask_output_project,
                                                with_name=with_name, prefix=now_name)


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)
    task.load_dataset(cfg.dataset.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    model = models[0]
    source_lang = cfg['task'].source_lang
    target_lang = cfg['task'].target_lang
    model.eval()
    model.patch_all_mask(source_lang, target_lang)
    mask_dict = {}
    if hasattr(model, "model_loop_iter"):
        itr = model.model_loop_iter(with_name=True)
    else:
        itr = loop_linear_module_for_model(model, no_mask_output_project=True, with_name=True)
    for name, m in itr:
        if name.startswith("soft_threshold"):
            continue
        mask_dict[name] = (m.weight != 0)
    torch_persistent_save(mask_dict, cfg['task'].dest)


def cli_main():
    parser = get_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


def get_parser():
    parser = options.get_generation_parser()
    group = parser.add_argument_group("generate_mask")
    group.add_argument("--dest", type=str)
    return parser


if __name__ == "__main__":
    cli_main()
