import argparse

import torch
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str,
                        help="The checkpoint path to generate the mask")
    parser.add_argument("--mask-path", type=str,
                        help="The output mask path")
    return parser.parse_args()


def random_like(x):
    x = x.float().new_ones(x.size()) * 0.5
    return torch.bernoulli(x).bool()


def main():
    args = get_args()
    checkpoint_path = args.checkpoint_path
    mask_path = args.mask_path

    with PathManager.open(checkpoint_path, "rb") as f:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

    mask_dict = {}
    for k, v in state['model'].items():
        if "weight" in k:
            mask_dict[k] = random_like(v)

    torch_persistent_save(mask_dict, mask_path)


if __name__ == '__main__':
    main()
