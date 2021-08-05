import argparse

import torch
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager
from more_itertools import flatten, collapse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str,
                        help="The checkpoint path to generate the mask")
    parser.add_argument("--mask-path", type=str,
                        help="The output mask path")
    parser.add_argument("--gen-mask-with-prob", action="store_true")
    parser.add_argument("--gen-random-mask", action="store_true")
    parser.add_argument("--mask-prob", type=float)
    parser.add_argument('--gen-part', type=str, choices=["encoder", "decoder", "all"], required=True)
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--exclude-output-proj",action="store_true")
    return parser.parse_args()



def gen_each_random_mask_with_prob(weight, p):
    """

    :param weight: a tensor
    :param p: probability
    :return:
    """
    mask = torch.rand_like(weight.float()) > p

    return mask


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
    gen_part = args.gen_part
    p = args.mask_prob
    if args.gen_mask_with_prob or args.gen_random_mask:
        mask_dict = gen_mask_with_prob(state, p, gen_part, random_gen=args.gen_random_mask,
                                       include_embedding=args.include_embedding,exclude_output_proj=args.exclude_output_proj)
    else:
        raise NotImplementedError

    torch_persistent_save(mask_dict, mask_path)


def gen_each_mask_with_prob(weight, p):
    """

    :param weight: a tensor
    :param p: probability
    :return:
    """
    total_size = weight.nelement()
    kth = int(p * total_size)
    weight = torch.abs(weight.float())
    kth_element, _ = torch.kthvalue(weight.view(-1), k=kth, dim=0)
    kth_element = kth_element.tolist()  # float
    mask = weight > kth_element

    return mask


def gen_embedding_mask_with_prob(weight, p):
    # mask embedding is a little different, we treat each vector in embedding weight as a linear.
    weight = torch.abs(weight.float())
    dim = weight.size(1)
    kth = int(p * dim)
    kth_element, _ = torch.kthvalue(weight, k=kth, dim=1, keepdim=True)  # kth_element: (num_vec,1)
    mask = weight > kth_element

    return mask


def gen_mask_with_prob(state, p, gen_part="all", random_gen=False, include_embedding=False,exclude_output_proj=False):
    """
    generate mask with probability
    :return: mask_dict
    """
    mask_dict = {}
    gen_func = gen_each_random_mask_with_prob if random_gen else gen_each_mask_with_prob
    for k, v in state['model'].items():
        if "weight" in k and "embed" not in k.lower() and "layer_norm" not in k:
            if gen_part == "all":
                mask_dict[k] = gen_func(v, p)
            elif gen_part == "encoder":
                if "encoder" in k:
                    mask_dict[k] = gen_func(v, p)
            elif gen_part == "decoder":
                if "decoder" in k:
                    mask_dict[k] = gen_func(v, p)
            else:
                raise NotImplementedError
        if include_embedding and "weight" in k and "embed" in k.lower():
            mask_dict[k] = gen_embedding_mask_with_prob(v, p)

    if exclude_output_proj:
        for k in list(state['model'].keys()):
            if 'output_projection' in k:
                print("Delete output projection")
                del state['model'][k]
    return mask_dict


if __name__ == '__main__':
    main()
