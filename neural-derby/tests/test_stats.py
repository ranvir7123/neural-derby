import torch.nn as nn
from derby.horses import blocks, stats, stable


def test_count_params_matches_torch():
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 8 * 8, 4))
    assert stats.count_params(m) == sum(p.numel() for p in m.parameters())


def test_count_macs_conv_and_linear_by_hand():
    # conv: 8 output maps of 8x8 = 512 outputs, each needs 3*3*3 = 27 multiplies -> 13824
    # linear: 512 inputs x 4 outputs -> 2048
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(512, 4))
    assert stats.count_macs(m, img=8) == 13824 + 2048


def test_count_macs_includes_transformer_attention_products():
    # A 96x96 image cut into 8x8 patches gives 12 x 12 = 144 tokens, each 64 numbers wide.
    # patch conv: 144 tokens x 64 outputs x (3 colours x 8 x 8 = 192 multiplies each) = 1,769,472
    # q, k, v, o: 4 linear layers, each 144 tokens x 64 outputs x 64 inputs -> 4 x 589,824 = 2,359,296
    # feed-forward: 144 x 128 outputs x 64 inputs, then 144 x 64 outputs x 128 inputs -> 2 x 1,179,648 = 2,359,296
    # attention products (no weights, so no layer sees them): q @ k^T and att @ v,
    #   each 144 x 144 x 64 multiplies -> 2 x 1,327,104 = 2,654,208
    m = nn.Sequential(blocks.PatchEmbed(3, 64, 8, 96), blocks.TransformerEncoderBlock(64, 4, 128))
    patch_conv = 144 * 64 * (3 * 8 * 8)
    projections = 4 * 144 * 64 * 64
    feed_forward = 144 * 64 * 128 + 144 * 128 * 64
    attention = 2 * 144 * 144 * 64
    assert patch_conv == 1_769_472 and attention == 2_654_208
    assert stats.count_macs(m) == patch_conv + projections + feed_forward + attention


def test_depthwise_has_fewer_macs_than_textbook():
    assert stats.count_macs(stable.build_featherweight()) < stats.count_macs(stable.build_textbook())


def test_depth_counts_weight_layers():
    m = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3), nn.Flatten(), nn.Linear(8, 2))
    assert stats.depth(m) == 3


def test_layer_table_rows_have_shapes_and_params():
    rows = stats.layer_table(stable.build_textbook())
    assert rows[0]["type"] == "Conv2d"
    assert rows[0]["out_shape"] == [24, 96, 96]
    assert rows[0]["params"] == 3 * 24 * 9
    assert rows[-1]["type"] == "Linear" and rows[-1]["out_shape"] == [6]


def test_horse_stats_bundle():
    s = stats.horse_stats(stable.by_key("textbook"))
    assert set(s) == {"params", "macs", "depth", "table"}
    assert s["params"] > 0 and s["macs"] > 0 and s["depth"] == 4


def test_layer_table_lists_each_weight_layer_once():
    m = stable.build_spotlight()
    rows = stats.layer_table(m)
    names = [r["layer"] for r in rows]
    assert len(names) == len(set(names)) == stats.depth(m)
