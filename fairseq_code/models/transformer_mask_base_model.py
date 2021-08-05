from abc import ABCMeta, abstractmethod

from fairseq.models.transformer import TransformerModel


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        if not prefix.endswith("."):
            submodule_prefix = prefix + ("." if prefix else "") + name
        else:
            submodule_prefix = prefix + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


class TransformerMaskBaseModel(TransformerModel, metaclass=ABCMeta):

    @staticmethod
    def add_args(parser):
        super(TransformerMaskBaseModel, TransformerMaskBaseModel).add_args(parser)

    @staticmethod
    def lang_pair(src: str, tgt: str):
        return f"{src}-{tgt}"

    def _format_name(self, k):
        return k.replace(".", "|")

    def upgrade_state_dict_named(self, state_dict, name):
        # Add the share parameter to the different name to match the pytorch api
        shared_names = [sorted(t) for t in _catalog_shared_params(self, prefix=name)]
        for names in shared_names:
            for _name in names[1:]:
                if names[0] not in state_dict:
                    continue
                if _name not in state_dict:
                    state_dict[_name] = state_dict[names[0]]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        shared_names = [sorted(t) for t in _catalog_shared_params(self, prefix=prefix)]
        for names in shared_names:
            for name in names[1:]:
                # Remove the shared parameter's other name
                del state_dict[name]
        return state_dict

    @abstractmethod
    def patch_all_mask(self, src_lang, tgt_lang):
        pass
