import io
from pathlib import Path

import pytest
from PIL import Image

from derby.data.live import LiveError, LiveStore


def jpeg_bytes(size=(120, 80), colour=(10, 200, 30), orientation=None) -> bytes:
    im = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    if orientation:
        exif = im.getexif()
        exif[274] = orientation
        im.save(buf, "JPEG", exif=exif.tobytes())
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def store(tmp_path: Path) -> LiveStore:
    return LiveStore(tmp_path / "live", caps={"per_device": 3, "total": 5, "max_bytes": 50_000})


def test_add_stores_photo_under_class_folder(store: LiveStore):
    p = store.add("dev_0001", "bottle", jpeg_bytes())
    assert p.path.exists() and p.path.parent.name == "bottle"
    assert p.label == 0 and store.count() == 1


def test_per_device_cap(store: LiveStore):
    for _ in range(3):
        store.add("dev_0001", "pen", jpeg_bytes())
    with pytest.raises(LiveError, match="3 photos"):
        store.add("dev_0001", "pen", jpeg_bytes())


def test_total_cap(store: LiveStore):
    for i in range(5):
        store.add(f"dev_{i:04d}", "book", jpeg_bytes())
    with pytest.raises(LiveError, match="full"):
        store.add("dev_9999", "book", jpeg_bytes())


def test_size_cap_and_bad_bytes_and_bad_class(store: LiveStore):
    with pytest.raises(LiveError, match="too big"):
        store.add("dev_0001", "shoe", b"x" * 60_000)
    with pytest.raises(LiveError, match="not a photo"):
        store.add("dev_0001", "shoe", b"definitely not an image")
    with pytest.raises(LiveError, match="six objects"):
        store.add("dev_0001", "laptop", jpeg_bytes())
    with pytest.raises(LiveError):
        store.add("../evil", "shoe", jpeg_bytes())


def test_exif_rotation_is_applied_on_save(store: LiveStore):
    p = store.add("dev_0001", "phone", jpeg_bytes(size=(100, 50), orientation=6))
    with Image.open(p.path) as im:
        assert im.size == (50, 100)


def test_split_by_device_never_leaks_a_device(store: LiveStore):
    store = LiveStore(store.root, caps={"per_device": 3, "total": 100, "max_bytes": 50_000})
    for d in range(6):
        for _ in range(2):
            store.add(f"dev_{d:04d}", "backpack", jpeg_bytes())
    train, test = store.split_by_device(seed=3)
    assert len(train) + len(test) == 12 and train and test
    assert {p.device for p in train}.isdisjoint({p.device for p in test})


def test_fewer_than_twelve_all_go_to_test(store: LiveStore):
    for d in range(4):
        store.add(f"dev_{d:04d}", "backpack", jpeg_bytes())
    train, test = store.split_by_device(seed=3)
    assert train == [] and len(test) == 4


def test_tensors_and_thumbnail(store: LiveStore):
    p = store.add("dev_0001", "shoe", jpeg_bytes())
    x, y = LiveStore.tensors([p], img=32)
    assert x.shape == (1, 3, 32, 32) and y.tolist() == [5]
    assert LiveStore.thumbnail_b64(p).startswith("/9j/")  # base64 of a JPEG starts like this
