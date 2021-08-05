#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import argparse

from fairseq import options

from .calculate_sad import get_calculate_sad_parser
from .util import get_all_task_options, load_config


def main_from_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target-path", type=str, required=True)
    args = parser.parse_args()
    config = args.config
    config = load_config(config)
    param_args = get_all_task_options(config)
    param_args.extend(
        ["--target-path",
         args.target_path,
         "--path",
         config["save_dir"]]
    )
    sad_parser = get_calculate_sad_parser()
    args = options.parse_args_and_arch(sad_parser, input_args=param_args)
    from .calculate_sad import main as sad_main
    sad_main(args)


if __name__ == "__main__":
    main_from_config()
