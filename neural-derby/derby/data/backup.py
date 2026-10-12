"""The safety-net dataset built from LVIS/COCO crops. Folder per class, label = index in config.CLASSES."""
import collections
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
    groups: list[str] | None = None  # source photo of each row, see group_key


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
    return Loaded(load_many(paths, img), y, paths, [group_key(p) for p in paths])


def split_indices(n: int, seed: int, test_frac: float = config.BACKUP_TEST_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]


def group_key(path: Path) -> str:
    """The photo a crop came from: the file name without its box number (coco_123_1 -> coco_123).

    A name with no _<number> ending (an old-style name like 0007) is returned as it is.
    """
    stem, sep, box = path.stem.rpartition("_")
    return stem if sep and box.isdigit() else path.stem


def split_by_group(groups: list[str], seed: int,
                   test_frac: float = config.BACKUP_TEST_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    """Split by source photo, so two crops of one photo never land one in train and one in test.

    Whole photos go to the test side, in a shuffled order fixed by the seed, until it holds at least
    round(n * test_frac) rows. Returns (train indices, test indices), each in ascending order.
    """
    rows_of = collections.defaultdict(list)
    for i, g in enumerate(groups):
        rows_of[g].append(i)
    names = sorted(rows_of)
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
    n_test = int(round(len(groups) * test_frac))
    test: list[int] = []
    for k in order:
        if len(test) >= n_test:
            break
        test.extend(rows_of[names[k]])
    test_set = set(test)
    train = [i for i in range(len(groups)) if i not in test_set]
    return torch.tensor(train, dtype=torch.long), torch.tensor(sorted(test), dtype=torch.long)
