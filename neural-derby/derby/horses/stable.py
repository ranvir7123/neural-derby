"""The seven horses. Each is one of ma'am's blocks with the same head: global average pool + one linear layer."""
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from derby import config
from derby.horses.blocks import (
    ChannelAttention, ConvBlock, DepthwiseSeparable, InceptionBlock, PatchEmbed,
    PlainPair, ResidualBlock, SpatialAttention, TransformerEncoderBlock,
)

N_CLASSES = len(config.CLASSES)


def head(c: int) -> nn.Sequential:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, N_CLASSES))


def pool() -> nn.Module:
    return nn.MaxPool2d(2)


def build_textbook() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        ConvBlock(24, 48), pool(),
        ConvBlock(48, 96), pool(),
        head(96),
    )


# The depth knob for STAIRS and ELEVATOR: (width, number of pairs) per stage.
# Both horses use exactly this; only the pair type differs. Tune in prep (Task 14).
DEEP_STAGES = [(24, 1), (48, 1), (96, 2)]


def _tower(skips: bool) -> nn.Module:
    layers: list[nn.Module] = [ConvBlock(3, 24)]
    prev = 24
    for width, pairs in DEEP_STAGES:
        if width != prev:
            layers.append(ConvBlock(prev, width))
            prev = width
        for _ in range(pairs):
            layers.append(ResidualBlock(width) if skips else PlainPair(width))
        layers.append(pool())
    return nn.Sequential(*layers, head(prev))


def build_stairs() -> nn.Module:
    return _tower(skips=False)


def build_elevator() -> nn.Module:
    return _tower(skips=True)


def build_featherweight() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        DepthwiseSeparable(24, 48), pool(),
        DepthwiseSeparable(48, 96), pool(),
        head(96),
    )


def build_spotlight() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        ConvBlock(24, 48), ChannelAttention(48), SpatialAttention(), pool(),
        ConvBlock(48, 96), ChannelAttention(96), SpatialAttention(), pool(),
        head(96),
    )


def build_trifocals() -> nn.Module:
    return nn.Sequential(
        ConvBlock(3, 24), pool(),
        InceptionBlock(24, 16), pool(),     # 48 channels out
        InceptionBlock(48, 32), pool(),     # 96 channels out
        head(96),
    )


class Influencer(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 3, heads: int = 4):
        super().__init__()
        self.embed = PatchEmbed(3, dim, 8, config.IMG)
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(dim, heads, dim * 2) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        tokens = self.blocks(self.embed(x))
        return self.fc(self.norm(tokens.mean(dim=1)))


def build_influencer() -> nn.Module:
    return Influencer()


@dataclass(frozen=True)
class Horse:
    key: str
    name: str
    tagline: str
    colour: str
    block_name: str
    build: Callable[[], nn.Module]


STABLE: list[Horse] = [
    Horse("textbook", "TEXTBOOK",
          "Conv, pool, repeat. The recipe every network starts from.",
          "#ffd23f", "plain conv + maxpool", build_textbook),
    Horse("stairs", "STAIRS",
          "11 floors deep and the learning signal has to climb every step. Tires early, strong finish.",
          "#ff6b35", "deep conv, no shortcuts", build_stairs),
    Horse("elevator", "ELEVATOR",
          "Same building, but with shortcuts. The signal rides straight through.",
          "#3fb8ff", "residual (skip connection)", build_elevator),
    Horse("featherweight", "FEATHERWEIGHT",
          "One cheap step per colour channel, then mix. 4.5 times fewer multiplications than TEXTBOOK.",
          "#7bff6b", "depthwise separable conv", build_featherweight),
    Horse("spotlight", "SPOTLIGHT",
          "Turns up the volume on useful features and points a light at where the object is.",
          "#ff3fd0", "channel + spatial attention", build_spotlight),
    Horse("trifocals", "TRIFOCALS",
          "Three lens sizes at once. Sees the whole shoe and the shoelace.",
          "#b56bff", "inception (multiscale)", build_trifocals),
    Horse("influencer", "THE INFLUENCER",
          "The architecture behind ChatGPT. Famous. Needs a million photos to perform.",
          "#ff4040", "transformer encoder", build_influencer),
]


def enabled_stable(enabled: dict[str, bool] | None = None) -> list[Horse]:
    enabled = config.HORSES_ENABLED if enabled is None else enabled
    return [h for h in STABLE if enabled.get(h.key, True)]


def by_key(key: str) -> Horse:
    for h in STABLE:
        if h.key == key:
            return h
    raise KeyError(key)
