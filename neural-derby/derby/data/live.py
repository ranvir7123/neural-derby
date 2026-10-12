"""Photos from the room's phones: validation, caps, EXIF rotation, per-device split."""
import base64
import io
import re
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from derby import config
from derby.data.images import load_many

DEVICE_RE = re.compile(r"[A-Za-z0-9_-]{4,32}")


@dataclass(frozen=True)
class Photo:
    device: str
    class_name: str
    path: Path

    @property
    def label(self) -> int:
        return config.CLASSES.index(self.class_name)


class LiveError(Exception):
    """A rejection whose message is safe to show on the phone."""


class LiveStore:
    def __init__(self, root: Path, caps: dict = config.CAPS, classes: list[str] = config.CLASSES, long_side: int = 160):
        self.root = Path(root)
        self.caps = caps
        self.classes = classes
        self.long_side = long_side
        self._photos: list[Photo] = []
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def count(self) -> int:
        return len(self._photos)

    def count_for(self, device: str) -> int:
        return sum(p.device == device for p in self._photos)

    def photos(self) -> list[Photo]:
        return list(self._photos)

    def add(self, device: str, class_name: str, data: bytes) -> Photo:
        # One upload at a time: counting, naming, saving and recording must not interleave.
        with self._lock:
            if not DEVICE_RE.fullmatch(device or ""):
                raise LiveError("Reload the page and try again.")
            if class_name not in self.classes:
                raise LiveError("Pick one of the six objects first.")
            if len(data) > self.caps["max_bytes"]:
                raise LiveError("That photo is too big. Try again with a smaller one.")
            if len(self._photos) >= self.caps["total"]:
                raise LiveError("The paddock is full. Thanks! Now watch the race.")
            if self.count_for(device) >= self.caps["per_device"]:
                raise LiveError(f"You have sent your {self.caps['per_device']} photos. Now place a bet!")
            try:
                probe = Image.open(io.BytesIO(data))
                probe.verify()
                im = Image.open(io.BytesIO(data))
                # Many phone cameras write MPO, a JPEG that holds more than one picture.
                if im.format not in ("JPEG", "MPO", "PNG"):
                    raise LiveError("Only JPEG or PNG photos, please.")
                im = ImageOps.exif_transpose(im).convert("RGB")
            except Image.DecompressionBombError:
                raise LiveError("That photo is too big. Try again with a smaller one.") from None
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError, struct.error):
                raise LiveError("That file is not a photo we can read.") from None
            im.thumbnail((self.long_side, self.long_side))
            folder = self.root / class_name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{device}_{self.count_for(device)}.jpg"
            im.save(path, "JPEG", quality=90)
            photo = Photo(device, class_name, path)
            self._photos.append(photo)
            return photo

    def split_by_device(self, seed: int, min_for_split: int = config.MIN_LIVE_FOR_SPLIT) -> tuple[list[Photo], list[Photo]]:
        """Half the devices go to test, half to train. Too few photos: everything is test."""
        photos = list(self._photos)
        if len(photos) < min_for_split:
            return [], photos
        devices = sorted({p.device for p in photos})
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(devices), generator=g).tolist()
        test_devices = {devices[i] for i in perm[: len(devices) // 2]}
        train = [p for p in photos if p.device not in test_devices]
        test = [p for p in photos if p.device in test_devices]
        return train, test

    @staticmethod
    def tensors(photos: list[Photo], img: int = config.IMG) -> tuple[torch.Tensor, torch.Tensor]:
        x = load_many([p.path for p in photos], img)
        y = torch.tensor([p.label for p in photos], dtype=torch.long)
        return x, y

    @staticmethod
    def thumbnail_b64(photo: Photo, size: int = 96) -> str:
        with Image.open(photo.path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
