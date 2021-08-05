import argparse

import torch
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager
from more_itertools import flatten, collapse
from pprint import pprint


# dict_keys(['encoder.layers.0.self_attn.k_proj.weight', 'encoder.layers.0.self_attn.v_proj.weight', 'encoder.layers.0.self_attn.q_proj.weight', 'encoder.layers.0.self_attn.out_proj.weight', 'encoder.layers.0.self_attn_layer_norm.weight', 'encoder.layers.0.fc1.weight', 'encoder.layers.0.fc2.weight', 'encoder.layers.0.final_layer_norm.weight', 'encoder.layers.1.self_attn.k_proj.weight', 'encoder.layers.1.self_attn.v_proj.weight', 'encoder.layers.1.self_attn.q_proj.weight', 'encoder.layers.1.self_attn.out_proj.weight', 'encoder.layers.1.self_attn_layer_norm.weight', 'encoder.layers.1.fc1.weight', 'encoder.layers.1.fc2.weight', 'encoder.layers.1.final_layer_norm.weight', 'encoder.layers.2.self_attn.k_proj.weight', 'encoder.layers.2.self_attn.v_proj.weight', 'encoder.layers.2.self_attn.q_proj.weight', 'encoder.layers.2.self_attn.out_proj.weight', 'encoder.layers.2.self_attn_layer_norm.weight', 'encoder.layers.2.fc1.weight', 'encoder.layers.2.fc2.weight', 'encoder.layers.2.final_layer_norm.weight', 'encoder.layers.3.self_attn.k_proj.weight', 'encoder.layers.3.self_attn.v_proj.weight', 'encoder.layers.3.self_attn.q_proj.weight', 'encoder.layers.3.self_attn.out_proj.weight', 'encoder.layers.3.self_attn_layer_norm.weight', 'encoder.layers.3.fc1.weight', 'encoder.layers.3.fc2.weight', 'encoder.layers.3.final_layer_norm.weight', 'encoder.layers.4.self_attn.k_proj.weight', 'encoder.layers.4.self_attn.v_proj.weight', 'encoder.layers.4.self_attn.q_proj.weight', 'encoder.layers.4.self_attn.out_proj.weight', 'encoder.layers.4.self_attn_layer_norm.weight', 'encoder.layers.4.fc1.weight', 'encoder.layers.4.fc2.weight', 'encoder.layers.4.final_layer_norm.weight', 'encoder.layers.5.self_attn.k_proj.weight', 'encoder.layers.5.self_attn.v_proj.weight', 'encoder.layers.5.self_attn.q_proj.weight', 'encoder.layers.5.self_attn.out_proj.weight', 'encoder.layers.5.self_attn_layer_norm.weight', 'encoder.layers.5.fc1.weight', 'encoder.layers.5.fc2.weight', 'encoder.layers.5.final_layer_norm.weight', 'decoder.layers.0.self_attn.k_proj.weight', 'decoder.layers.0.self_attn.v_proj.weight', 'decoder.layers.0.self_attn.q_proj.weight', 'decoder.layers.0.self_attn.out_proj.weight', 'decoder.layers.0.self_attn_layer_norm.weight', 'decoder.layers.0.encoder_attn.k_proj.weight', 'decoder.layers.0.encoder_attn.v_proj.weight', 'decoder.layers.0.encoder_attn.q_proj.weight', 'decoder.layers.0.encoder_attn.out_proj.weight', 'decoder.layers.0.encoder_attn_layer_norm.weight', 'decoder.layers.0.fc1.weight', 'decoder.layers.0.fc2.weight', 'decoder.layers.0.final_layer_norm.weight', 'decoder.layers.1.self_attn.k_proj.weight', 'decoder.layers.1.self_attn.v_proj.weight', 'decoder.layers.1.self_attn.q_proj.weight', 'decoder.layers.1.self_attn.out_proj.weight', 'decoder.layers.1.self_attn_layer_norm.weight', 'decoder.layers.1.encoder_attn.k_proj.weight', 'decoder.layers.1.encoder_attn.v_proj.weight', 'decoder.layers.1.encoder_attn.q_proj.weight', 'decoder.layers.1.encoder_attn.out_proj.weight', 'decoder.layers.1.encoder_attn_layer_norm.weight', 'decoder.layers.1.fc1.weight', 'decoder.layers.1.fc2.weight', 'decoder.layers.1.final_layer_norm.weight', 'decoder.layers.2.self_attn.k_proj.weight', 'decoder.layers.2.self_attn.v_proj.weight', 'decoder.layers.2.self_attn.q_proj.weight', 'decoder.layers.2.self_attn.out_proj.weight', 'decoder.layers.2.self_attn_layer_norm.weight', 'decoder.layers.2.encoder_attn.k_proj.weight', 'decoder.layers.2.encoder_attn.v_proj.weight', 'decoder.layers.2.encoder_attn.q_proj.weight', 'decoder.layers.2.encoder_attn.out_proj.weight', 'decoder.layers.2.encoder_attn_layer_norm.weight', 'decoder.layers.2.fc1.weight', 'decoder.layers.2.fc2.weight', 'decoder.layers.2.final_layer_norm.weight', 'decoder.layers.3.self_attn.k_proj.weight', 'decoder.layers.3.self_attn.v_proj.weight', 'decoder.layers.3.self_attn.q_proj.weight', 'decoder.layers.3.self_attn.out_proj.weight', 'decoder.layers.3.self_attn_layer_norm.weight', 'decoder.layers.3.encoder_attn.k_proj.weight', 'decoder.layers.3.encoder_attn.v_proj.weight', 'decoder.layers.3.encoder_attn.q_proj.weight', 'decoder.layers.3.encoder_attn.out_proj.weight', 'decoder.layers.3.encoder_attn_layer_norm.weight', 'decoder.layers.3.fc1.weight', 'decoder.layers.3.fc2.weight', 'decoder.layers.3.final_layer_norm.weight', 'decoder.layers.4.self_attn.k_proj.weight', 'decoder.layers.4.self_attn.v_proj.weight', 'decoder.layers.4.self_attn.q_proj.weight', 'decoder.layers.4.self_attn.out_proj.weight', 'decoder.layers.4.self_attn_layer_norm.weight', 'decoder.layers.4.encoder_attn.k_proj.weight', 'decoder.layers.4.encoder_attn.v_proj.weight', 'decoder.layers.4.encoder_attn.q_proj.weight', 'decoder.layers.4.encoder_attn.out_proj.weight', 'decoder.layers.4.encoder_attn_layer_norm.weight', 'decoder.layers.4.fc1.weight', 'decoder.layers.4.fc2.weight', 'decoder.layers.4.final_layer_norm.weight', 'decoder.layers.5.self_attn.k_proj.weight', 'decoder.layers.5.self_attn.v_proj.weight', 'decoder.layers.5.self_attn.q_proj.weight', 'decoder.layers.5.self_attn.out_proj.weight', 'decoder.layers.5.self_attn_layer_norm.weight', 'decoder.layers.5.encoder_attn.k_proj.weight', 'decoder.layers.5.encoder_attn.v_proj.weight', 'decoder.layers.5.encoder_attn.q_proj.weight', 'decoder.layers.5.encoder_attn.out_proj.weight', 'decoder.layers.5.encoder_attn_layer_norm.weight', 'decoder.layers.5.fc1.weight', 'decoder.layers.5.fc2.weight', 'decoder.layers.5.final_layer_norm.weight', 'decoder.output_projection.weight'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask1-path", type=str)
    parser.add_argument("--mask2-path", type=str)
    parser.add_argument("--which-part", choices=['encoder', 'decoder', 'all'])
    parser.add_argument("--which-layer", default=None, choices=["0", "1", "2", "3", "4", "5"], help="start from 0")
    parser.add_argument("--query", nargs='+',
                        choices=["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.out_proj",
                                 "encoder_attn.q", "encoder_attn.k", "encoder_attn.v", "encoder_attn.out_proj",
                                 "fc"], help="input query for specific component")
    parser.add_argument("--include-output-project", action="store_true")
    return parser.parse_args()


def _cal_similarity(mask1_weight, mask2_weight):
    w1_size = mask1_weight.numel()
    w2_size = mask2_weight.numel()

    assert w1_size == w2_size, "Size mismatch"

    total_one_num = (mask1_weight == 1).sum().item()
    overlap_one = (mask1_weight & mask2_weight).sum().item()

    return {"total": total_one_num, "overlap": overlap_one}


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
    query = args.query
    part = args.which_part
    layer = args.which_layer
    mask1_path = args.mask1_path
    mask2_path = args.mask2_path
    print("calculate similarity with {part} in layer {layer}, the query {query}".format(part=part, layer=layer,
                                                                                        query=query))

    mask1_state = load_mask(mask1_path)
    mask2_state = load_mask(mask2_path)

    # calculate
    total_cnt = 0
    cnt = 0
    layer_cond = "" if layer is None else "layers." + layer
    part_cond = "" if part == "all" else part + "."
    print(layer_cond, part_cond)
    for k, v in mask1_state.items():
        if k not in mask2_state:
            raise ValueError
        if "layer_norm" in k:
            continue
        if "projection" in k and not args.include_output_project:
            continue

        result = None
        if layer_cond in k and part_cond in k:
            if query is not None:
                for q in query:
                    if q in k:
                        print(k)
                        result = _cal_similarity(v, mask2_state[k])
                        break
            else:
                print(k)
                result = _cal_similarity(v, mask2_state[k])

            if result is not None:
                total_cnt += result['total']
                cnt += result['overlap']

    print("Total Cnt is {total}, Overlap Cnt is {overlap}, the similarity of {path1} and {path2} is {p:.2f}%".format(
        total=total_cnt, overlap=cnt, path1=mask1_path, path2=mask2_path, p=cnt / total_cnt * 100))


if __name__ == '__main__':
    main()

