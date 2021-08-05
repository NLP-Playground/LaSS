import math
import os
import re

import yaml

import argparse

from fairseq_code.utils import file_operation
from fairseq_code.utils.common import command


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", type=str)
    parser.add_argument("--force-reset", action="store_true")
    return parser.parse_args()


def get_user_dir_path():
    d = file_operation.path_split(__file__)[0]
    d = file_operation.path_split(d)[0]
    return file_operation.join_paths(d, "fairseq_code")


def pop_from_config(config, key):
    if key in config:
        return config.pop(key)
    key = key.replace("_", "-")
    if key in config:
        return config.pop(key)
    key = key.replace("-", "_")
    if key in config:
        return config.pop(key)
    return None


def _get_environment_variable(name: str):
    return os.environ.get(name)


def no_restore_if_checkpoint_last_exits(config, force_reset):
    save_dir = config['save_dir'] if 'save_dir' in config else config['save-dir']
    if file_operation.file_exists(
        file_operation.join_paths(
            save_dir, "checkpoint_last.pt"
        )
    ):
        if "restore-file" in config:
            del config["restore-file"]
        if "restore_file" in config:
            del config["restore_file"]
        if not force_reset:
            for key in list(config.keys()):
                if key.startswith("reset"):
                    del config[key]


def train(args):
    config_paths = args.config
    force_reset = args.force_reset
    config = {}
    for config_path in config_paths:
        with file_operation.open_file(config_path, "r") as f:
            config = {**yaml.safe_load(f), **config}

    no_restore_if_checkpoint_last_exits(config, force_reset)
    args = []
    for k, v in config.items():
        if k == "data_bin":
            args.append(v)
            continue
        k = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                k = f"--{k}"
            else:
                k = ""
        else:
            k = f"--{k} {v}"
        args.append(k)
    args = " ".join(args)

    command(
        f"fairseq-train --user-dir {get_user_dir_path()} {args}"
    )


if __name__ == "__main__":
    train(get_args())
