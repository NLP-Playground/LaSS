import re

import torch
from torch import nn as nn


def loop_linear_module_for_model(m, no_mask_output_project=False, with_name=False, prefix="",
                                    skip_pattern=None):

    if skip_pattern is not None:
        pattern = re.compile(skip_pattern)
    else:
        pattern = None
    for n, c in m.named_children():

        if no_mask_output_project and n == "output_projection":
            continue

        now_name = n if len(prefix) == 0 else prefix + "." + n

        if skip_pattern is not None:
            if now_name == skip_pattern:
                continue
            if pattern.match(now_name) is not None:
                continue

        if isinstance(c, nn.Linear):
            if with_name:
                yield now_name, c
            else:
                yield c
        yield from loop_linear_module_for_model(c, no_mask_output_project=no_mask_output_project,
                                                with_name=with_name, prefix=now_name, skip_pattern=skip_pattern)


def get_row_mask(weight):
    return torch.all(weight == 0, dim=1)


