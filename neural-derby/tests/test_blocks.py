import torch
from derby.horses import blocks as B

X = torch.randn(2, 3, 96, 96)


def test_conv_block_keeps_size_changes_channels():
    assert B.ConvBlock(3, 24)(X).shape == (2, 24, 96, 96)
    assert B.ConvBlock(3, 24, k=5)(X).shape == (2, 24, 96, 96)


def test_plain_pair_and_residual_same_shape():
    x = torch.randn(2, 24, 48, 48)
    assert B.PlainPair(24)(x).shape == x.shape
    assert B.ResidualBlock(24)(x).shape == x.shape


def test_channel_attention_scales_each_channel_between_0_and_1():
    x = torch.rand(2, 32, 12, 12) + 0.1
    y = B.ChannelAttention(32)(x)
    ratio = y / x
    assert torch.all(ratio >= 0) and torch.all(ratio <= 1)
    # the same channel gets the same scale at every pixel
    assert torch.allclose(ratio[0, 0].flatten(), ratio[0, 0].flatten()[0].expand(144), atol=1e-5)


def test_spatial_attention_masks_every_channel_the_same():
    x = torch.rand(2, 32, 12, 12) + 0.1
    y = B.SpatialAttention()(x)
    ratio = y / x
    assert torch.all(ratio >= 0) and torch.all(ratio <= 1)
    assert torch.allclose(ratio[0, 0], ratio[0, 5], atol=1e-5)


def test_inception_concatenates_three_branches():
    x = torch.randn(2, 24, 48, 48)
    assert B.InceptionBlock(24, 16)(x).shape == (2, 48, 48, 48)


def test_depthwise_separable_shape():
    x = torch.randn(2, 24, 48, 48)
    assert B.DepthwiseSeparable(24, 48)(x).shape == (2, 48, 48, 48)


def test_patch_embed_and_transformer_block():
    tokens = B.PatchEmbed(3, 64, 8, 96)(X)
    assert tokens.shape == (2, 144, 64)
    assert B.TransformerEncoderBlock(64, 4, 128)(tokens).shape == (2, 144, 64)
