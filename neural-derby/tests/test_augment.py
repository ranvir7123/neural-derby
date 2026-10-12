import base64

import torch

from derby.data import augment


def batch(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 96, 96), generator=g, dtype=torch.uint8)


def test_to_float_and_normalize_ranges():
    x = augment.to_float(batch())
    assert x.dtype == torch.float32 and x.min() >= 0 and x.max() <= 1
    z = augment.normalize(x)
    assert z.min() >= -1 and z.max() <= 1


def test_augment_shape_range_and_determinism():
    x = batch()
    a = augment.augment(x, torch.Generator().manual_seed(1))
    b = augment.augment(x, torch.Generator().manual_seed(1))
    c = augment.augment(x, torch.Generator().manual_seed(2))
    assert a.shape == x.shape and a.dtype == torch.float32
    assert a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_augment_actually_changes_the_image():
    x = batch()
    a = augment.augment(x, torch.Generator().manual_seed(1))
    assert (a - augment.to_float(x)).abs().mean() > 0.05


def test_preview_pairs_are_png_base64():
    pairs = augment.preview_pairs(batch(6), n=4, seed=0)
    assert len(pairs) == 4
    for orig, aug in pairs:
        assert base64.b64decode(orig)[:8] == b"\x89PNG\r\n\x1a\n"
        assert base64.b64decode(aug)[:8] == b"\x89PNG\r\n\x1a\n"
