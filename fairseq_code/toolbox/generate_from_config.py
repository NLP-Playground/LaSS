"""
This script will evaluate the model.

The generate.log will be saved at the ${save_dir}/results/${checkpoint_name}/${src}-${tgt}/generate.log

"""


import argparse
import io
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy

from fairseq import options
from tqdm import tqdm

from fairseq_code.utils import file_operation
from .util import load_config, get_all_task_options


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_last.pt")
    parser.add_argument("--lang-pairs", type=str, default=None,
                        help="THe format is the same as fairseq multilingual translaiton task")
    parser.add_argument("--evaluate-bin", type=str, default=None,
                        help="The bin to evaluate")
    parser.add_argument("--count-inference-flops", action="store_true",
                        help="Count the inference flops")
    return parser.parse_known_args()


def main():
    args, generate_args = get_args()
    config = args.config
    checkpoint_name = args.checkpoint_name
    lang_pairs = args.lang_pairs
    count_inference_flops = args.count_inference_flops

    config = load_config(config)

    if args.evaluate_bin is not None:
        config['data_bin'] = args.evaluate_bin

    param_args = get_all_task_options(config)
    checkpoint_path = file_operation.join_paths(
        config["save_dir"],
        checkpoint_name
    )

    if lang_pairs is None:
        lang_pairs = config['lang_pairs']
    lang_pairs = lang_pairs.strip().split(",")

    for lang_pair in tqdm(lang_pairs):
        src, tgt = lang_pair.strip().split("-")

        generate_log = io.StringIO()

        with redirect_stdout(generate_log), redirect_stderr(generate_log):
            generate(checkpoint_path, param_args, src=src, tgt=tgt, count_inference_flops=count_inference_flops,
                     other_args=generate_args)

        generate_log.seek(0)
        generate_log = generate_log.read()
        print(generate_log)
        if count_inference_flops:
            lang_pair += "-flops"
        with file_operation.open_file(
                file_operation.join_paths(
                    config["save_dir"],
                    "results",
                    checkpoint_name,
                    lang_pair,
                    "generate.log"
                ),
                "w"
        ) as f:
            f.write(generate_log)


def generate(checkpoint_path, param_args, src, tgt, count_inference_flops=False, other_args=None):
    param_args = deepcopy(param_args)
    param_args.extend(
        [
            "--path",
            checkpoint_path,
            "-s", src,
            "-t", tgt,
        ]
    )
    if other_args is not None:
        param_args.extend(other_args)
    if other_args is not None and "--max-tokens" not in other_args:
        param_args.extend(["--max-tokens", "12000"])
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=param_args)
    if count_inference_flops:
        from .generate_to_count_flops import main as generate_main
    else:
        from fairseq_cli.generate import main as generate_main
    generate_main(args)


if __name__ == '__main__':
    main()
