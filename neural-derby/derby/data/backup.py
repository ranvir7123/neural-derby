"""The safety-net dataset built from LVIS/COCO crops. Folder per class, label = index in config.CLASSES."""
from dataclasses import dataclass
from pathlib import Path

import torch

from derby import config
from derby.data.images import load_many

EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Loaded:
    x: torch.Tensor          # uint8, N x 3 x img x img
    y: torch.Tensor          # long, N
    paths: list[Path]


def list_class_files(root: Path) -> list[tuple[Path, int]]:
    out = []
    for label, cls in enumerate(config.CLASSES):
        folder = root / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append((p, label))
    return out


def load(root: Path = config.DATA_DIR / "backup", img: int = config.IMG) -> Loaded:
    files = list_class_files(root)
    paths = [p for p, _ in files]
    y = torch.tensor([label for _, label in files], dtype=torch.long)
    return Loaded(load_many(paths, img), y, paths)


def split_indices(n: int, seed: int, test_frac: float = config.BACKUP_TEST_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]
