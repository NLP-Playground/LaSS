import argparse

import yaml
from fairseq import tasks
from fairseq.file_io import PathManager


def get_all_task_options(config):
    task = tasks.get_task(config['task'])
    parser = argparse.ArgumentParser()
    task.add_args(parser)
    option_names = list(parser._option_string_actions.keys())
    option_names = filter(lambda x: x.startswith("--"), option_names)
    option_names = map(lambda x: x.lstrip("-"), option_names)
    option_names = list(option_names)
    option_names += ["bpe", "tokenizer"]
    param_args = [config["data_bin"], "--task", config['task']]
    for option_name in option_names:
        key_in_config = option_name.replace("-", "_")
        if key_in_config not in config:
            continue
        option_value = config[key_in_config]
        if isinstance(option_value, bool):
            if option_value:
                param_args.append(
                    f"--{option_name}"
                )
        else:
            param_args.append(
                f"--{option_name}"
            )
            param_args.append(
                str(option_value)
            )
    return param_args


def load_config(config_path):
    with PathManager.open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config