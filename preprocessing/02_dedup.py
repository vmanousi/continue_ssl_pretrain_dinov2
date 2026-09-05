"""
Perceptual-hash near-duplicate detection over the final (cropped/resized)
SSL corpus produced by 01_process_images.py.

HyperKvasir's authors state the data "has not been pre-processed or
augmented in any way" -- so near-duplicates are not assumed to be already
removed. This runs the cheap pHash pass we agreed to try before considering
a heavier DINOv2-embedding semantic dedup pass.

Method: 64-bit perceptual hash (imagehash.phash) per image, grouped by
Hamming distance <= --threshold via union-find. Within each duplicate group,
keeps one representative (prefers "labeled" source, then higher mean
luminance, then lowest image_id for determinism) and marks the rest as
duplicates -- non-destructive: no source files are touched, only a new
manifest is written.

Writes:
  <output-dir>/dedup_log.csv       image_id, phash, dup_group_id, is_representative
  <output-dir>/train_deduped.txt   final manifest for the DINOv2 dataloader (representatives only)
"""
import argparse
import csv
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


POPCOUNT_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def popcount64(arr_uint64):
    return POPCOUNT_TABLE[arr_uint64.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def find_near_duplicate_pairs(hash_ints, threshold, num_bands=8):
    """
    LSH banding over the 64-bit hash: split into `num_bands` non-overlapping
    byte-sized bands. With num_bands > threshold, pigeonhole guarantees any
    pair within `threshold` Hamming distance shares at least one identical
    band exactly -- so it's enough to compare candidates within each band's
    buckets, never the full n x n matrix. (threshold=6, num_bands=8 -> exact
    recall for distance <= 7.)

    Returns a list of (i, j) index pairs with true Hamming distance <= threshold.
    """
    assert num_bands > threshold, "num_bands must exceed threshold for guaranteed recall"
    n = len(hash_ints)
    band_bytes = hash_ints.view(np.uint8).reshape(n, 8)  # one byte per band

    found_pairs = set()
    for band in range(num_bands):
        band_values = band_bytes[:, band]
        order = np.argsort(band_values, kind="stable")
        sorted_values = band_values[order]
        # contiguous runs of equal value = buckets
        boundaries = np.nonzero(np.diff(sorted_values))[0] + 1
        buckets = np.split(order, boundaries)
        for bucket in buckets:
            if len(bucket) < 2:
                continue
            bucket_hashes = hash_ints[bucket]
            # small m x m Hamming distance matrix via broadcasting, m = bucket size
            xor = bucket_hashes[:, None] ^ bucket_hashes[None, :]
            dist = popcount64(xor.ravel()).reshape(xor.shape)
            ii, jj = np.nonzero((dist <= threshold) & (dist >= 0))
            for a, b in zip(ii, jj):
                if bucket[a] < bucket[b]:
                    found_pairs.add((int(bucket[a]), int(bucket[b])))
    return found_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--preprocessing-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=3, help="max Hamming distance to consider a near-duplicate (of 64 bits) -- 3 chosen after visual audit: threshold=6 showed false positives (different content, similar low-frequency structure), threshold=3 was 8/8 clean on a visual sample")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.preprocessing_log) as f:
        rows = [r for r in csv.DictReader(f) if r["kept"] == "True"]
    if args.limit:
        rows = rows[: args.limit]

    t0 = time.time()
    print(f"Computing pHash for {len(rows)} images...")
    hashes = []
    for i, row in enumerate(rows):
        path = args.images_dir / f"{row['image_id']}.jpg"
        with Image.open(path) as im:
            h = imagehash.phash(im, hash_size=8)  # 64-bit
        hashes.append(int(str(h), 16))
        if (i + 1) % 20000 == 0:
            print(f"  hashed {i + 1}/{len(rows)}")

    t_hash_done = time.time()
    print(f"  hashing done in {t_hash_done - t0:.1f}s")

    hash_ints = np.array(hashes, dtype=np.uint64)

    print("Finding near-duplicate candidate pairs via LSH banding...")
    pairs = find_near_duplicate_pairs(hash_ints, args.threshold, num_bands=8)
    print(f"  found {len(pairs)} pairs within Hamming distance {args.threshold} "
          f"in {time.time() - t_hash_done:.1f}s")

    uf = UnionFind(len(rows))
    for i, j in pairs:
        uf.union(i, j)

    groups = {}
    for i in range(len(rows)):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    def rep_key(idx):
        row = rows[idx]
        source_rank = 0 if row["source"] == "labeled" else 1
        lum = float(row["mean_luminance"]) if row["mean_luminance"] else 0.0
        return (source_rank, -lum, row["image_id"])

    dedup_rows = []
    kept_ids = []
    for group_id, indices in groups.items():
        indices_sorted = sorted(indices, key=rep_key)
        representative_idx = indices_sorted[0]
        for idx in indices:
            is_rep = idx == representative_idx
            dedup_rows.append(
                {
                    "image_id": rows[idx]["image_id"],
                    "phash": format(hashes[idx], "016x"),
                    "dup_group_id": group_id,
                    "group_size": len(indices),
                    "is_representative": is_rep,
                }
            )
            if is_rep:
                kept_ids.append(rows[idx]["image_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "dedup_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "phash", "dup_group_id", "group_size", "is_representative"])
        writer.writeheader()
        writer.writerows(dedup_rows)

    with open(args.output_dir / "train_deduped.txt", "w") as f:
        for image_id in sorted(kept_ids):
            f.write(f"images/{image_id}.jpg\n")

    dup_groups = [g for g in groups.values() if len(g) > 1]
    removed = len(rows) - len(kept_ids)
    print("\n" + "=" * 60)
    print("DEDUP SUMMARY")
    print("=" * 60)
    print(f"Input images:        {len(rows)}")
    print(f"Duplicate groups (size > 1): {len(dup_groups)}")
    print(f"Images removed:      {removed} ({100 * removed / len(rows):.2f}%)")
    print(f"Final corpus size:   {len(kept_ids)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
