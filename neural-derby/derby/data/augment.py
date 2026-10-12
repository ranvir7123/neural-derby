"""Module 4 on the GPU: one affine warp (crop, rotate, flip), brightness and contrast, noise, one erased box.

Everything here works on a whole batch at once with torch ops, so a lap stays fast.
The generator passed in must be on the same device as the images.
"""
import base64
import io
import math

import torch
import torch.nn.functional as F
from PIL import Image


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float().div_(255.0)


def normalize(x_float: torch.Tensor) -> torch.Tensor:
    return (x_float - 0.5) / 0.5


def _uniform(shape, lo: float, hi: float, gen: torch.Generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=device) * (hi - lo) + lo


def augment(x_uint8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    x = to_float(x_uint8)
    b = x.shape[0]
    dev = x.device

    # 1. crop + rotate + flip as one affine warp
    scale = _uniform((b,), 0.6, 1.0, gen, dev)                       # 1.0 = whole image, 0.6 = zoomed crop
    angle = _uniform((b,), -15.0, 15.0, gen, dev) * math.pi / 180.0
    flip = torch.where(torch.rand((b,), generator=gen, device=dev) < 0.5, -1.0, 1.0)
    tx = _uniform((b,), -1.0, 1.0, gen, dev) * (1.0 - scale)
    ty = _uniform((b,), -1.0, 1.0, gen, dev) * (1.0 - scale)
    cos, sin = torch.cos(angle), torch.sin(angle)
    row0 = torch.stack([scale * cos * flip, -scale * sin, tx], dim=1)
    row1 = torch.stack([scale * sin * flip, scale * cos, ty], dim=1)
    theta = torch.stack([row0, row1], dim=1)                          # b x 2 x 3
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    # 2. brightness (add) and contrast (multiply around the middle grey)
    contrast = _uniform((b, 1, 1, 1), 0.7, 1.3, gen, dev)
    brightness = _uniform((b, 1, 1, 1), -0.3, 0.3, gen, dev)
    x = (x - 0.5) * contrast + 0.5 + brightness

    # 3. noise
    x = x + torch.randn(x.shape, generator=gen, device=dev) * 0.02

    # 4. erase one rectangle on 30 percent of the images
    x = _erase(x, gen, p=0.3)
    return x.clamp_(0.0, 1.0)


def _erase(x: torch.Tensor, gen: torch.Generator, p: float) -> torch.Tensor:
    b, _, h, w = x.shape
    dev = x.device
    do = torch.rand((b,), generator=gen, device=dev) < p
    eh = (_uniform((b,), 0.2, 0.5, gen, dev) * h).long()
    ew = (_uniform((b,), 0.2, 0.5, gen, dev) * w).long()
    y0 = (torch.rand((b,), generator=gen, device=dev) * (h - eh)).long()
    x0 = (torch.rand((b,), generator=gen, device=dev) * (w - ew)).long()
    ys = torch.arange(h, device=dev).view(1, h, 1)
    xs = torch.arange(w, device=dev).view(1, 1, w)
    inside = (ys >= y0.view(b, 1, 1)) & (ys < (y0 + eh).view(b, 1, 1)) \
        & (xs >= x0.view(b, 1, 1)) & (xs < (x0 + ew).view(b, 1, 1))
    mask = (inside & do.view(b, 1, 1)).unsqueeze(1)
    return x.masked_fill(mask, 0.0)


def _png_b64(x_float_3hw: torch.Tensor) -> str:
    arr = (x_float_3hw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview_pairs(x_uint8_cpu: torch.Tensor, n: int = 48, seed: int = 0) -> list[tuple[str, str]]:
    """Real before/after pairs for the MULTIPLY act. Runs on CPU on a sample."""
    x = x_uint8_cpu[:n].cpu()
    aug = augment(x, torch.Generator().manual_seed(seed))
    orig = to_float(x)
    return [(_png_b64(orig[i]), _png_b64(aug[i])) for i in range(x.shape[0])]
