"""Turn a photo file into the fixed-size uint8 tensor every horse eats."""
from pathlib import Path

import torch
from PIL import Image, ImageOps

from derby import config


def open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)   # phones store rotation as a tag; apply it
    return im.convert("RGB")


def to_square(im: Image.Image, img: int) -> Image.Image:
    w, h = im.size
    scale = img / min(w, h)
    im = im.resize((max(img, round(w * scale)), max(img, round(h * scale))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - img) // 2, (h - img) // 2
    return im.crop((left, top, left + img, top + img))


def load_image(path: Path, img: int = config.IMG) -> torch.Tensor:
    im = to_square(open_rgb(path), img)
    arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
    return arr.view(img, img, 3).permute(2, 0, 1).contiguous()


def load_many(paths: list[Path], img: int = config.IMG) -> torch.Tensor:
    if not paths:
        return torch.empty(0, 3, img, img, dtype=torch.uint8)
    return torch.stack([load_image(p, img) for p in paths])
