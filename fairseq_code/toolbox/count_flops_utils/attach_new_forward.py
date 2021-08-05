import types
from typing import Dict

# from fvcore.nn import FlopCountAnalysis
# import fvcore
from fvcore import nn as fv_nn
from torch import Tensor
import torch

from .module_wrapper import TracingAdapter


_IGNORED_OPS = {
    "aten::add",
    "aten::add_",
    "aten::argmax",
    "aten::argsort",
    "aten::batch_norm",
    "aten::constant_pad_nd",
    "aten::div",
    "aten::div_",
    "aten::exp",
    "aten::log2",
    "aten::max_pool2d",
    "aten::meshgrid",
    "aten::mul",
    "aten::mul_",
    "aten::neg",
    "aten::nonzero_numpy",
    "aten::reciprocal",
    "aten::rsub",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::softmax",
    "aten::sort",
    "aten::sqrt",
    "aten::sub",
    "torchvision::nms",
}


class FlopCountAnalysis(fv_nn.FlopCountAnalysis):
    """
    Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
    """

    def __init__(self, model, inputs):
        """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
        """
        wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
        super().__init__(wrapper, wrapper.flattened_inputs)
        self.set_op_handle(**{k: None for k in _IGNORED_OPS})


@torch.no_grad()
def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
    inputs = [sample]
    for k in ["prefix_tokens", "constraints", "bos_token"]:
        if k in kwargs:
            inputs.append(kwargs[k])
        else:
            inputs.append(None)
    inputs = tuple(inputs)
    flops = FlopCountAnalysis(self, inputs)
    return flops


def attach_new_generate_method_to_generator(generator):
    generator.generate = types.MethodType(
        generate,
        generator
    )

def count_flops(model, **kwargs):
    inputs = []
    for k in ["src_tokens", "src_lengths", "prev_output_tokens", 
            "return_all_hiddens", "features_only", "alignment_layer", "alignment_heads"]:
        if k in kwargs:
            inputs.append(kwargs[k])
        else:
            inputs.append(None)
    inputs = tuple(inputs)
    flops = FlopCountAnalysis(model, inputs)
    return flops