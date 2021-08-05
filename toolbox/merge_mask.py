import argparse

import torch
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager
from more_itertools import flatten, collapse
from pprint import pprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-mask", type=str)
    parser.add_argument("--decoder-mask", type=str)
    parser.add_argument("--output-mask", type=str)

    return parser.parse_args()


def load_mask(mask_path):
    with PathManager.open(mask_path, "rb") as f:
        mask_state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
    return mask_state


def main():
    args = get_args()
    pprint(args)
    encoder_mask_state = load_mask(args.encoder_mask)
    decoder_mask_state = load_mask(args.decoder_mask)

    output_mask_dict = {}
    for k, v in encoder_mask_state.items():
        if "encoder." in k:
            print(k)
            output_mask_dict[k] = v

    for k, v in decoder_mask_state.items():
        if "decoder." in k:
            print(k)
            output_mask_dict[k] = v

    torch_persistent_save(output_mask_dict, args.output_mask)


if __name__ == '__main__':
    main()
