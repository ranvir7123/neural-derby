from pathlib import Path

import torch
from PIL import Image

from derby import config
from derby.data import backup, images


def test_load_image_is_square_uint8(tiny_backup: Path):
    t = images.load_image(tiny_backup / "bottle" / "a.jpg", 96)
    assert t.dtype == torch.uint8 and t.shape == (3, 96, 96)


def test_load_image_applies_exif_rotation(tmp_path: Path):
    # a 100x50 image tagged as rotated 90 degrees must load as portrait before cropping
    p = tmp_path / "rot.jpg"
    im = Image.new("RGB", (100, 50), (255, 0, 0))
    im.paste((0, 0, 255), (0, 0, 50, 50))          # left half blue
    exif = im.getexif()
    exif[274] = 6                                  # orientation 6 = rotate 90 clockwise
    im.save(p, "JPEG", exif=exif.tobytes())
    t = images.load_image(p, 32)
    # after rotating clockwise the blue half is on top
    top = t[:, :8, :].float().mean(dim=(1, 2))
    assert top[2] > top[0]


def test_load_backup_labels_follow_class_order(tiny_backup: Path):
    loaded = backup.load(tiny_backup, img=32)
    assert loaded.x.shape == (12, 3, 32, 32)
    assert loaded.y.tolist() == [i for i in range(6) for _ in range(2)]
    assert len(loaded.paths) == 12


def test_split_is_deterministic_and_disjoint():
    a_train, a_test = backup.split_indices(100, seed=7, test_frac=0.15)
    b_train, b_test = backup.split_indices(100, seed=7, test_frac=0.15)
    assert torch.equal(a_train, b_train) and torch.equal(a_test, b_test)
    assert len(a_test) == 15 and len(a_train) == 85
    assert set(a_train.tolist()).isdisjoint(a_test.tolist())
