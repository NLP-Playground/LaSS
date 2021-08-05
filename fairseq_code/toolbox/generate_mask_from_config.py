"""
This script will evaluate the model.

The generate.log will be saved at the ${save_dir}/results/${checkpoint_name}/${src}-${tgt}/generate.log

"""


import argparse
from copy import deepcopy

from fairseq import options

from fairseq_code.utils import file_operation
from .util import load_config, get_all_task_options
from .generate_mask_from_softthreshold import get_parser
from .generate_mask_from_softthreshold import main as generate_mask_main


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_last.pt")
    parser.add_argument("--lang-pair", type=str, default=None,
                        help="THe format is the same as fairseq multilingual translaiton task")
    parser.add_argument("--target-path", type=str, required=True,
                        help="The target path of the mask")
    return parser.parse_args()


def main():
    args = get_args()
    config = args.config
    checkpoint_name = args.checkpoint_name
    lang_pair = args.lang_pair

    config = load_config(config)
    param_args = get_all_task_options(config)
    checkpoint_path = file_operation.join_paths(
        config["save_dir"],
        checkpoint_name
    )

    src, tgt = lang_pair.strip().split("-")
    generate(checkpoint_path, param_args, src=src, tgt=tgt,
             target_path=args.target_path)


def generate(checkpoint_path, param_args, src, tgt, target_path):
    param_args = deepcopy(param_args)
    param_args.extend(
        [
            "--path",
            checkpoint_path,
            "-s", src,
            "-t", tgt,
            "--dest", target_path,
        ]
    )
    parser = get_parser()
    args = options.parse_args_and_arch(parser, input_args=param_args)
    generate_mask_main(args)


if __name__ == '__main__':
    main()
