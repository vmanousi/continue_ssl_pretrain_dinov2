"""
Build a single manifest CSV over the full HyperKvasir corpus (labeled + unlabeled).

Read-only: does not move, copy, or modify any source image. This is the single
source of truth every later preprocessing step reads from and writes back into
(never re-walks the raw directories).

Usage:
    python 00_build_manifest.py \
        --labeled-root ../data/hyperkvasir_raw_labeled/labeled-images \
        --unlabeled-root ../data/hyperkvasir_raw_unlabeled/unlabeled-images/images \
        --output ../data/manifest_00.csv
"""
import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image


def iter_labeled(labeled_root: Path):
    """labeled-images/<upper|lower>-gi-tract/<category>/<subcategory...>/<uuid>.jpg"""
    for path in sorted(labeled_root.rglob("*.jpg")):
        class_path = path.relative_to(labeled_root).parent.as_posix()
        yield path, class_path


def iter_unlabeled(unlabeled_root: Path):
    for path in sorted(unlabeled_root.rglob("*.jpg")):
        yield path, ""


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-root", required=True, type=Path)
    parser.add_argument("--unlabeled-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    sources = [
        ("labeled", iter_labeled(args.labeled_root)),
        ("unlabeled", iter_unlabeled(args.unlabeled_root)),
    ]

    for source, iterator in sources:
        count = 0
        for path, class_path in iterator:
            try:
                with Image.open(path) as im:
                    width, height = im.size
                    mode = im.mode
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not open {path}: {exc}")
                continue

            rows.append(
                {
                    "image_id": path.stem,
                    "path": str(path.resolve()),
                    "source": source,
                    "class_path": class_path,
                    "orig_width": width,
                    "orig_height": height,
                    "orig_mode": mode,
                    "md5": md5sum(path),
                }
            )
            count += 1
            if count % 10000 == 0:
                print(f"  {source}: {count} processed...")
        print(f"{source}: {count} images total")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Exact-duplicate check across the whole corpus (by content hash).
    md5_to_ids = {}
    for row in rows:
        md5_to_ids.setdefault(row["md5"], []).append(row["image_id"])
    dup_groups = {k: v for k, v in md5_to_ids.items() if len(v) > 1}

    print(f"\nTotal images in manifest: {len(rows)}")
    print(f"Exact-duplicate groups (by md5): {len(dup_groups)}")
    if dup_groups:
        dup_count = sum(len(v) - 1 for v in dup_groups.values())
        print(f"  -> {dup_count} redundant exact-duplicate files")
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()
