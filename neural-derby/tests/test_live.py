import io
import threading
from concurrent.futures import ThreadPoolExecutor
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


def image_bytes(fmt: str, size=(120, 80), colour=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, fmt)
    return buf.getvalue()


def test_png_upload_is_accepted(store: LiveStore):
    p = store.add("dev_0001", "book", image_bytes("PNG"))
    assert p.path.exists() and store.count() == 1


def test_gif_upload_is_rejected_with_a_phone_safe_message(store: LiveStore):
    with pytest.raises(LiveError, match="Only JPEG or PNG"):
        store.add("dev_0001", "book", image_bytes("GIF"))
    assert store.count() == 0


def test_mpo_phone_photo_is_accepted(store: LiveStore):
    # Many phone cameras write MPO (a JPEG that holds more than one picture).
    im = Image.new("RGB", (120, 80), (10, 200, 30))
    buf = io.BytesIO()
    im.save(buf, "MPO", save_all=True, append_images=[im.copy()])
    data = buf.getvalue()
    assert Image.open(io.BytesIO(data)).format == "MPO"  # prove what we feed in
    p = store.add("dev_0001", "bottle", data)
    assert p.path.exists() and store.count() == 1


def test_overlapping_uploads_from_one_device_never_lose_or_duplicate_a_photo(tmp_path: Path):
    store = LiveStore(tmp_path / "live", caps={"per_device": 3, "total": 100, "max_bytes": 50_000})
    data = jpeg_bytes()
    start_together = threading.Barrier(6)

    def upload(_):
        start_together.wait(timeout=10)
        try:
            return store.add("dev_0001", "pen", data)
        except LiveError as err:
            return err

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(upload, range(6)))
    saved = [r for r in results if not isinstance(r, LiveError)]
    refused = [r for r in results if isinstance(r, LiveError)]
    assert len(saved) == 3 and len(refused) == 3
    assert store.count_for("dev_0001") == 3 and store.count() == 3
    assert len({p.path for p in saved}) == 3 and all(p.path.exists() for p in saved)


def test_decompression_bomb_becomes_a_live_error(store: LiveStore, monkeypatch: pytest.MonkeyPatch):
    data = jpeg_bytes(size=(100, 100))  # 10,000 pixels
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # more than twice over, so Pillow raises
    with pytest.raises(LiveError, match="too big"):
        store.add("dev_0001", "shoe", data)
    assert store.count() == 0
