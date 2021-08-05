#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import json
import logging
import os
import re
import sys
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import typing
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.file_io import PathManager
from omegaconf import DictConfig

from ..models.utils import get_row_mask


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


def loop_directory(directory: str) -> typing.List[str]:
    names = PathManager.ls(directory)
    pattern = re.compile("checkpoint_.*_(.*).pt")
    names = map(lambda x: (x, pattern.match(x)), names)
    names = filter(lambda x: x[1] is not None, names)
    names = sorted(names, key=lambda x: int(x[1].group(1)))
    return [f"{directory}/{name[0]}" for name in names]


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

    lang_pairs = task.args.lang_pairs
    # Load ensemble
    cached_weights = None
    sad_list = defaultdict(list)
    for checkpoint_path in loop_directory(cfg.common_eval.path):
        logger.info("loading model(s) from {}".format(checkpoint_path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            [checkpoint_path],
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        model = models[0]
        if use_cuda:
            model = model.cuda()
        model.eval()

        if cached_weights is None:
            prev_none = True
            cached_weights = defaultdict(list)
        else:
            prev_none = False
        for lang_pair in lang_pairs:
            new_weights = []
            src_lang, tgt_lang = lang_pair.split("-")
            model.patch_all_mask(src_lang=src_lang, tgt_lang=tgt_lang)
            for linear_module in model.model_loop_iter():
                new_weights.append(linear_module.weight)
            if not prev_none:
                sad = 0
                for old_weight, now_weight in zip(cached_weights[lang_pair], new_weights):
                    old_mask = get_row_mask(old_weight)
                    now_mask = get_row_mask(now_weight)
                    sad += torch.abs(old_mask.int()-now_mask.int()).sum()
                sad_list[lang_pair].append(sad.item())

            cached_weights[lang_pair] = new_weights

    with PathManager.open(cfg.task.target_path, "w") as f:
        json.dump(sad_list, f)


def cli_main():
    parser = get_calculate_sad_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


def get_calculate_sad_parser():
    parser = options.get_generation_parser()
    parser.add_argument("--target-path", type=str, help="The path to final sad list ")
    return parser


if __name__ == "__main__":
    cli_main()
