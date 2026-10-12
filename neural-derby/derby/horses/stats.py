"""Numbers ma'am asks for in exams: parameters, multiplies, depth, and a per-layer table."""
import torch
import torch.nn as nn

from derby import config
from derby.horses.blocks import TransformerEncoderBlock
from derby.horses.stable import Horse

WEIGHT_LAYERS = (nn.Conv2d, nn.Linear)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_macs(m: nn.Module, img: int = config.IMG) -> int:
    """Multiply-accumulates for one image, counting Conv2d and Linear layers.

    Conv: every output value needs (in_channels / groups) * kh * kw multiplies.
    Linear: every output value needs in_features multiplies.
    The transformer's two attention matrix products are counted too, through a hook on TransformerEncoderBlock
    (2 * T * T * dim per block); the Linear hooks alone would miss them.
    """
    total = 0
    hooks = []

    def conv_hook(mod, _inp, out):
        nonlocal total
        per_output = (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        total += (out.numel() // out.shape[0]) * per_output

    def linear_hook(mod, _inp, out):
        nonlocal total
        total += (out.numel() // out.shape[0]) * mod.in_features

    def attention_hook(_mod, inp, _out):
        nonlocal total
        _batch, tokens, dim = inp[0].shape
        total += 2 * tokens * tokens * dim

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
        elif isinstance(mod, TransformerEncoderBlock):
            hooks.append(mod.register_forward_hook(attention_hook))
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, img, img))
    m.train(was_training)
    for h in hooks:
        h.remove()
    return total


def depth(m: nn.Module) -> int:
    return sum(isinstance(mod, WEIGHT_LAYERS) for mod in m.modules())


def layer_table(m: nn.Module, img: int = config.IMG) -> list[dict]:
    """One row per weight layer, in forward order: layer name, type, output shape (no batch), params.

    A layer that runs more than once in a forward pass (ChannelAttention.score runs twice) gets one row,
    from its first run, so its params are not counted twice. Layers without conv or linear weights
    (BatchNorm and so on) have no row, so the rows add up to a little less than count_params.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    hooks = []

    def make_hook(name):
        def hook(mod, _inp, out):
            if name in seen:
                return
            seen.add(name)
            rows.append({
                "layer": name,
                "type": type(mod).__name__,
                "out_shape": list(out.shape[1:]),
                "params": sum(p.numel() for p in mod.parameters()),
            })
        return hook

    for name, mod in m.named_modules():
        if isinstance(mod, WEIGHT_LAYERS):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 3, img, img))
    m.train(was_training)
    for h in hooks:
        h.remove()
    return rows


def horse_stats(horse: Horse, img: int = config.IMG) -> dict:
    m = horse.build()
    return {
        "params": count_params(m),
        "macs": count_macs(m, img),
        "depth": depth(m),
        "table": layer_table(m, img),
    }
