"""Build data/backup/<class>/ from LVIS v1 annotations over COCO 2017 images. Run once, with internet.

Usage (from neural-derby/):  python tools/build_backup_dataset.py
Downloads only the photos it needs (about 0.5 GB). Safe to re-run with any options: every box has
a fixed file name, so a box that is already on disk is never saved again, and --per-class counts
the files already in the folder.
Licence: LVIS annotations CC BY 4.0; COCO images under Flickr terms (research use).

How a crop is cut: the object's box is grown into a square around its centre before cropping, so
thin objects such as pens keep their whole shape plus some of the desk around them. The size rule
(--min-side, or MIN_SIDE for pen and shoe) is checked on the object's own box, so tiny objects are
still left out. Within one photo and class, a box whose square mostly overlaps a crop already kept
(MAX_OVERLAP) is skipped: LVIS sometimes labels one object twice (as "book" and "notebook"), and
side-by-side objects in a small photo can grow into the same square.

Last stage: a class listed in OPEN_IMAGES_TOPUP that is still short after LVIS is topped up from the
Open Images test split, then its validation split (no account needed). It exists because LVIS has
few large pens. Those pens are often clean product-style photos on plain backgrounds, unlike the
busy COCO scenes the other classes come from. That stage downloads a 77 MB and a 25 MB box list
plus one photo per crop.
Licence: Open Images annotations CC BY 4.0; its images CC BY 2.0 (Flickr).

File names: coco_<image id>_<k>.jpg and oi_<split>_<image id>_<k>.jpg, where k is the box's
position among that photo's boxes of that class in the annotation file (counted before any filter).
"""
import argparse
import collections
import csv
import hashlib
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
# smaller objects allowed for classes that are short at the default --min-side
MIN_SIDE = {"pen": 48, "shoe": 48}
# two crops of one photo whose squares overlap more than this (intersection / union) count as one picture
MAX_OVERLAP = 0.7

# our class -> Open Images class name, for classes LVIS cannot fill
OPEN_IMAGES_TOPUP = {"pen": "Pen"}
OI_SPLITS = ["test", "validation"]
OI_CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BOXES_URL = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")  # a cut-off download never looks finished
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return dest
        except Exception as e:  # network hiccup: wait and retry
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def square_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Grow a box into a square of side max(w, h) around its centre.

    If the square sticks out past an edge it slides back inside the photo (the object stays whole);
    it is only cut short when the photo itself is narrower than the square.
    """
    side = max(w, h)
    left = min(max(0.0, x + w / 2 - side / 2), max(0.0, img_w - side))
    top = min(max(0.0, y + h / 2 - side / 2), max(0.0, img_h - side))
    return round(left), round(top), round(min(img_w, left + side)), round(min(img_h, top + side))


def overlap(a: tuple, b: tuple) -> float:
    """Shared area over combined area of two (left, top, right, bottom) boxes: 1 = same picture."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    shared = w * h
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


def is_near_copy(square: tuple, kept: list) -> bool:
    return any(overlap(square, s) > MAX_OVERLAP for s in kept)


def save_crop(im: Image.Image, square: tuple, dest: Path, long_side: int) -> None:
    crop = im.crop(square)
    crop.thumbnail((long_side, long_side))
    crop.save(dest, "JPEG", quality=90)


def count_crops(folder: Path) -> int:
    return len(list(folder.glob("*.jpg")))


def count_duplicates(folder: Path) -> int:
    seen, dupes = set(), 0
    for p in sorted(folder.glob("*.jpg")):
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        dupes += digest in seen
        seen.add(digest)
    return dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--min-side", type=int, default=64, help="for classes not listed in MIN_SIDE")
    ap.add_argument("--max-per-image", type=int, default=2)
    ap.add_argument("--long-side", type=int, default=160)
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "backup")
    args = ap.parse_args()

    cache = config.DATA_DIR / "cache"
    print("1/5 fetching LVIS annotations (350 MB zip)...")
    zpath = fetch(LVIS_URL, cache / "lvis_v1_train.json.zip")
    with zipfile.ZipFile(zpath) as z:
        lvis = json.load(z.open("lvis_v1_train.json"))

    name_to_group = {n: g for g, names in GROUPS.items() for n in names}
    cat_to_group = {c["id"]: name_to_group[c["name"]] for c in lvis["categories"] if c["name"] in name_to_group}
    images = {im["id"]: im for im in lvis["images"]}

    print("2/5 choosing LVIS boxes...")
    on_disk = {}
    for g in GROUPS:
        (args.out / g).mkdir(parents=True, exist_ok=True)
        on_disk[g] = {p.name for p in (args.out / g).glob("*.jpg")}
    # every box of our classes, in file order, with its fixed file name and its square
    candidates = []
    seen: dict[tuple, int] = collections.Counter()  # (photo, class) -> boxes met so far, gives k
    for a in lvis["annotations"]:
        g = cat_to_group.get(a["category_id"])
        if g is None:
            continue
        key = (a["image_id"], g)
        photo = images[a["image_id"]]
        square = square_box(*a["bbox"], photo["width"], photo["height"])
        candidates.append((g, key, f"coco_{a['image_id']}_{seen[key]}.jpg", a["bbox"], square))
        seen[key] += 1
    kept: dict[tuple, list] = collections.defaultdict(list)  # (photo, class) -> squares kept, old and new
    for g, key, name, _, square in candidates:
        if name in on_disk[g]:
            kept[key].append(square)
    picked: dict[str, list] = collections.defaultdict(list)
    near_copies: dict[str, int] = collections.Counter()
    for g, key, name, (x, y, w, h), square in candidates:
        if name in on_disk[g] or len(on_disk[g]) + len(picked[g]) >= args.per_class:
            continue
        if min(w, h) < MIN_SIDE.get(g, args.min_side) or len(kept[key]) >= args.max_per_image:
            continue
        if is_near_copy(square, kept[key]):
            near_copies[g] += 1
            continue
        kept[key].append(square)
        picked[g].append((key[0], name, square))
    for g in GROUPS:
        print(f"   {g}: {len(on_disk[g])} already on disk, {len(picked[g])} new, "
              f"{near_copies[g]} near-copies skipped")

    print("3/5 downloading photos and cropping...")
    for g, items in picked.items():
        for i, (img_id, name, square) in enumerate(items):
            url = images[img_id]["coco_url"]
            src = fetch(url, cache / "coco" / Path(url).name)
            save_crop(Image.open(src).convert("RGB"), square, args.out / g / name, args.long_side)
            if i % 50 == 0:
                print(f"   {g} {i}/{len(items)}")

    top_up_from_open_images(args, cache)

    print("5/5 done. Counts:")
    dupes = 0
    for g in GROUPS:
        d = count_duplicates(args.out / g)
        dupes += d
        print(f"   {g}: {count_crops(args.out / g)}" + (f" ({d} duplicate files)" if d else ""))
    print(f"duplicates: {dupes}")


def open_images_boxes(boxes_csv: Path, label: str) -> dict[str, list]:
    """Read the big box list one row at a time; keep single, real objects of one class, grouped by photo.

    k numbers each box among its photo's boxes of this class before any filter, so a box always
    gets the same file name.
    """
    by_photo, seen = collections.defaultdict(list), collections.Counter()
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] != label:
                continue
            k = seen[row["ImageID"]]
            seen[row["ImageID"]] += 1
            if row["IsGroupOf"] == "0" and row["IsDepiction"] == "0":
                by_photo[row["ImageID"]].append((k, *(float(row[c]) for c in ("XMin", "XMax", "YMin", "YMax"))))
    return by_photo


def top_up_from_open_images(args, cache: Path) -> None:
    print("4/5 Open Images top-up...")
    short = [g for g in OPEN_IMAGES_TOPUP if count_crops(args.out / g) < args.per_class]
    if not short:
        print("   not needed.")
        return
    oi_cache = cache / "openimages"
    with open(fetch(OI_CLASSES_URL, oi_cache / Path(OI_CLASSES_URL).name), newline="", encoding="utf-8") as f:
        label_of = {row["DisplayName"]: row["LabelName"] for row in csv.DictReader(f)}

    for g in short:
        out_dir = args.out / g
        min_side = MIN_SIDE.get(g, args.min_side)
        for split in OI_SPLITS:
            have = count_crops(out_dir)
            if have >= args.per_class:
                break
            boxes_url = OI_BOXES_URL.format(split=split)
            boxes_csv = fetch(boxes_url, oi_cache / Path(boxes_url).name)
            by_photo = open_images_boxes(boxes_csv, label_of[OPEN_IMAGES_TOPUP[g]])
            need = args.per_class - have
            print(f"   {g} ({split} split): {sum(map(len, by_photo.values()))} single real boxes in "
                  f"{len(by_photo)} photos; {have} on disk, adding up to {need}")
            added = near_copies = 0
            for img_id, photo_boxes in by_photo.items():
                if added >= need:
                    break
                dests = [out_dir / f"oi_{split}_{img_id}_{k}.jpg" for k, *_ in photo_boxes]
                n_old = sum(d.exists() for d in dests)
                if n_old == len(dests) or n_old >= args.max_per_image:
                    continue
                try:
                    url = OI_IMAGE_URL.format(split=split, image_id=img_id)
                    src = fetch(url, oi_cache / split / f"{img_id}.jpg")
                    im = Image.open(src).convert("RGB")
                except (RuntimeError, OSError) as e:  # missing or broken photo: skip it, keep going
                    print(f"   skipped {img_id}: {e}")
                    continue
                img_w, img_h = im.size
                boxes = [(x0 * img_w, y0 * img_h, (x1 - x0) * img_w, (y1 - y0) * img_h)
                         for _, x0, x1, y0, y1 in photo_boxes]
                squares = [square_box(*b, img_w, img_h) for b in boxes]
                kept = [s for s, d in zip(squares, dests) if d.exists()]
                for box, square, dest in zip(boxes, squares, dests):
                    if added >= need or len(kept) >= args.max_per_image:
                        break
                    if dest.exists() or min(box[2], box[3]) < min_side:
                        continue
                    if is_near_copy(square, kept):
                        near_copies += 1
                        continue
                    save_crop(im, square, dest, args.long_side)
                    kept.append(square)
                    if added % 50 == 0:
                        print(f"   {g} ({split} split) {added}/{need}")
                    added += 1
            print(f"   {g} ({split} split): added {added}, {near_copies} near-copies skipped")


if __name__ == "__main__":
    main()
