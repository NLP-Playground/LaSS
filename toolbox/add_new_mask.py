import torch

import argparse

import torch
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager
from more_itertools import flatten, collapse
from pprint import pprint
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-mask-name", type=str, help="like en-zh")
    parser.add_argument("--new-mask-path", type=str)
    parser.add_argument("--input-ckp", type=str)
    parser.add_argument("--output-ckp", type=str)

    return parser.parse_args()


def main():
    args = get_args()
    pprint(args)
    new_mask_name = args.new_mask_name
    new_mask_path = args.new_mask_path
    model = torch.load(args.input_ckp)

    mask_path_dict = json.loads(model['cfg']['model'].mask_path)
    if new_mask_name in mask_path_dict:
        raise ValueError

    mask_path_dict[new_mask_name] = new_mask_path

    mask_str = json.dumps(mask_path_dict)

    model['cfg']['model'].mask_path = mask_str

    torch_persistent_save(model, args.output_ckp)


if __name__ == '__main__':
    main()
