from pathlib import Path

import pytest
from PIL import Image

from derby import config


def make_image(path: Path, colour, size=(200, 150)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")


@pytest.fixture
def tiny_backup(tmp_path: Path) -> Path:
    """Two images per class, solid colours, landscape and portrait mixed."""
    root = tmp_path / "backup"
    for i, cls in enumerate(config.CLASSES):
        shade = 40 * i
        make_image(root / cls / "a.jpg", (shade, 100, 200), (200, 150))
        make_image(root / cls / "b.jpg", (200, shade, 100), (120, 180))
    return root
