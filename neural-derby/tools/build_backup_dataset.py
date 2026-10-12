"""Build data/backup/<class>/ from LVIS v1 annotations over COCO 2017 images. Run once, with internet.

Usage (from neural-derby/):  python tools/build_backup_dataset.py --per-class 400 --min-side 64
Downloads only the photos it needs (about 0.5 GB). Safe to re-run; it skips what exists.
Licence: LVIS annotations CC BY 4.0; COCO images under Flickr terms (research use).

Last stage: a class listed in OPEN_IMAGES_TOPUP that is still short after LVIS is topped up from
the Open Images test split (no account needed). It exists because LVIS has few large pens.
That stage also downloads a 77 MB box list plus one photo per crop.
Licence: Open Images annotations CC BY 4.0; its images CC BY 2.0 (Flickr).
"""
import argparse
import collections
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derby import config  # noqa: E402

LVIS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
GROUPS = {
    "bottle": {"water_bottle", "bottle"},
    "pen": {"pen", "pencil"},
    "book": {"book", "notebook"},
    "phone": {"cellular_telephone"},
    "backpack": {"backpack"},
    "shoe": {"shoe"},
}

# our class -> Open Images class name, for classes LVIS cannot fill
OPEN_IMAGES_TOPUP = {"pen": "Pen"}
OI_CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BOXES_URL = "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/test/{}.jpg"
OI_SOURCES = "openimages_sources.txt"  # one line per Open Images crop: file, image id, box


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:  # network hiccup: wait and retry
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--long-side", type=int, default=160)
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "backup")
    args = ap.parse_args()

    cache = config.DATA_DIR / "cache"
    print("1/4 fetching LVIS annotations (350 MB zip)...")
    zpath = fetch(LVIS_URL, cache / "lvis_v1_train.json.zip")
    with zipfile.ZipFile(zpath) as z:
        lvis = json.load(z.open("lvis_v1_train.json"))

    name_to_group = {n: g for g, names in GROUPS.items() for n in names}
    cat_to_group = {c["id"]: name_to_group[c["name"]] for c in lvis["categories"] if c["name"] in name_to_group}
    images = {im["id"]: im for im in lvis["images"]}

    print("2/4 choosing boxes...")
    picked: dict[str, list] = collections.defaultdict(list)
    per_image: dict[tuple, int] = collections.Counter()
    for a in lvis["annotations"]:
        g = cat_to_group.get(a["category_id"])
        if g is None or len(picked[g]) >= args.per_class:
            continue
        x, y, w, h = a["bbox"]
        if min(w, h) < args.min_side or per_image[(a["image_id"], g)] >= args.max_per_image:
            continue
        per_image[(a["image_id"], g)] += 1
        picked[g].append((a["image_id"], x, y, w, h))
    for g in GROUPS:
        print(f"   {g}: {len(picked[g])} crops")

    print("3/4 downloading photos and cropping...")
    for g, items in picked.items():
        out_dir = args.out / g
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (img_id, x, y, w, h) in enumerate(items):
            dest = out_dir / f"{i:04d}.jpg"
            if dest.exists():
                continue
            url = images[img_id]["coco_url"]
            src = fetch(url, cache / "coco" / Path(url).name)
            im = Image.open(src).convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
            im.thumbnail((args.long_side, args.long_side))
            im.save(dest, "JPEG", quality=90)
            if i % 50 == 0:
                print(f"   {g} {i}/{len(items)}")

    print("4/4 done. Counts:")
    for g in GROUPS:
        print(f"   {g}: {len(list((args.out / g).glob('*.jpg')))}")

    top_up_from_open_images(args, cache)


def count_crops(folder: Path) -> int:
    return len(list(folder.glob("*.jpg")))


def open_images_boxes(boxes_csv: Path, label: str) -> list[tuple[str, str, str, str, str]]:
    """Read the big box list one row at a time; keep single, real objects of one class."""
    keep = []
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] == label and row["IsGroupOf"] == "0" and row["IsDepiction"] == "0":
                keep.append((row["ImageID"], row["XMin"], row["XMax"], row["YMin"], row["YMax"]))
    return keep


def top_up_from_open_images(args, cache: Path) -> None:
    short = [g for g in OPEN_IMAGES_TOPUP if count_crops(args.out / g) < args.per_class]
    if not short:
        print("Open Images top-up: not needed.")
        return
    oi_cache = cache / "openimages"
    print("Open Images top-up: fetching class list and test-split boxes (77 MB)...")
    with open(fetch(OI_CLASSES_URL, oi_cache / Path(OI_CLASSES_URL).name), newline="", encoding="utf-8") as f:
        label_of = {row["DisplayName"]: row["LabelName"] for row in csv.DictReader(f)}
    boxes_csv = fetch(OI_BOXES_URL, oi_cache / Path(OI_BOXES_URL).name)

    for g in short:
        out_dir = args.out / g
        out_dir.mkdir(parents=True, exist_ok=True)
        boxes = open_images_boxes(boxes_csv, label_of[OPEN_IMAGES_TOPUP[g]])
        # the sources file lists boxes already cropped, so a re-run never saves the same box twice
        sources = out_dir / OI_SOURCES
        done = []
        if sources.exists():
            done = [line.split() for line in sources.read_text(encoding="utf-8").splitlines()]
        used = {tuple(parts[1:]) for parts in done}
        per_image = collections.Counter(parts[1] for parts in done)
        need = args.per_class - count_crops(out_dir)
        next_index = max((int(p.stem) for p in out_dir.glob("*.jpg") if p.stem.isdigit()), default=-1) + 1
        print(f"   {g}: {len(boxes)} single real boxes in the test split; adding up to {need} crops")

        added = 0
        for box in boxes:
            if added >= need:
                break
            img_id, x0, x1, y0, y1 = box
            if box in used or per_image[img_id] >= args.max_per_image:
                continue
            try:
                src = fetch(OI_IMAGE_URL.format(img_id), oi_cache / "test" / f"{img_id}.jpg")
                im = Image.open(src).convert("RGB")
            except (RuntimeError, OSError) as e:  # missing or broken photo: skip it, keep going
                print(f"   skipped {img_id}: {e}")
                continue
            w, h = im.size
            left, top, right, bottom = (round(float(x0) * w), round(float(y0) * h),
                                        round(float(x1) * w), round(float(y1) * h))
            if min(right - left, bottom - top) < args.min_side:
                continue
            crop = im.crop((left, top, right, bottom))
            crop.thumbnail((args.long_side, args.long_side))
            dest = out_dir / f"{next_index:04d}.jpg"
            crop.save(dest, "JPEG", quality=90)
            with open(sources, "a", encoding="utf-8") as f:
                f.write(" ".join([dest.name, *box]) + "\n")
            used.add(box)
            per_image[img_id] += 1
            if added % 50 == 0:
                print(f"   {g} {added}/{need}")
            next_index += 1
            added += 1

    print("Open Images top-up done. Counts:")
    for g in GROUPS:
        print(f"   {g}: {count_crops(args.out / g)}")


if __name__ == "__main__":
    main()
