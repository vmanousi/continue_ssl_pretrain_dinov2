"""
Verification audit: does pHash (02_dedup.py) miss near-duplicates that a
DINOv2 embedding (semantic) comparison would catch? EndoDINO uses DINOv2
embeddings for exactly this reason -- viewpoint/lighting-shifted near-dupes
that differ enough in raw pixels to fool a perceptual hash, but which a
model trained to understand "what is this" still recognizes as the same
thing.

Method: sample N images from the final deduped corpus, embed each with a
generic (not yet domain-adapted) DINOv2 ViT-S/14+reg, find each image's
nearest neighbor by cosine similarity, and check whether that neighbor was
ALSO caught by pHash. Pairs with high cosine similarity that pHash did NOT
flag are the interesting case -- written out as a contact sheet for visual
judgment (genuine miss vs. embedding-space false positive from generically
similar mucosa).

Does not modify anything -- read-only diagnostic.
"""
import argparse
import csv
import random
from pathlib import Path

import imagehash
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    model.eval()
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def hamming(h1, h2):
    return bin(h1 ^ h2).count("1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--train-list", required=True, type=Path, help="train_deduped.txt")
    parser.add_argument("--dedup-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--cosine-threshold", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.train_list) as f:
        all_ids = [line.strip().split("/")[-1].replace(".jpg", "") for line in f if line.strip()]

    random.seed(args.seed)
    sample_ids = random.sample(all_ids, min(args.sample_size, len(all_ids)))
    print(f"Sampled {len(sample_ids)} images from the deduped corpus")

    # phash Hamming distance already computed by 02_dedup.py -- reuse it
    # instead of recomputing, so this stays a comparison against the SAME
    # pHash values already used to build the corpus.
    phash_by_id = {}
    with open(args.dedup_log) as f:
        for row in csv.DictReader(f):
            phash_by_id[row["image_id"]] = int(row["phash"], 16)

    print("Loading DINOv2 ViT-S/14+reg (generic, not domain-adapted)...")
    model = load_model()
    transform = build_transform()

    print("Computing embeddings...")
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sample_ids), 64):
            batch_ids = sample_ids[i : i + 64]
            tensors = []
            for image_id in batch_ids:
                im = Image.open(args.images_dir / f"{image_id}.jpg").convert("RGB")
                tensors.append(transform(im))
            batch = torch.stack(tensors)
            feats = model(batch)  # (B, embed_dim) CLS token
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.numpy())
            if (i + 64) % 640 == 0:
                print(f"  embedded {min(i + 64, len(sample_ids))}/{len(sample_ids)}")

    embeddings = np.concatenate(embeddings, axis=0)  # (N, D), L2-normalized

    print("Computing pairwise cosine similarity and nearest neighbors...")
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -1.0)  # exclude self-match

    nearest_idx = similarity.argmax(axis=1)
    nearest_sim = similarity.max(axis=1)

    rows_out = []
    missed_by_phash = []
    for i, image_id in enumerate(sample_ids):
        j = nearest_idx[i]
        neighbor_id = sample_ids[j]
        sim = float(nearest_sim[i])
        if sim < args.cosine_threshold:
            continue
        h1, h2 = phash_by_id.get(image_id), phash_by_id.get(neighbor_id)
        phash_distance = hamming(h1, h2) if h1 is not None and h2 is not None else None
        caught_by_phash = phash_distance is not None and phash_distance <= 3
        rows_out.append(
            {
                "image_id": image_id, "neighbor_id": neighbor_id,
                "cosine_similarity": f"{sim:.4f}", "phash_hamming_distance": phash_distance,
                "caught_by_phash": caught_by_phash,
            }
        )
        if not caught_by_phash:
            missed_by_phash.append((image_id, neighbor_id, sim, phash_distance))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "semantic_dedup_audit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "neighbor_id", "cosine_similarity",
                                                "phash_hamming_distance", "caught_by_phash"])
        writer.writeheader()
        writer.writerows(rows_out)

    print("\n" + "=" * 60)
    print("SEMANTIC DEDUP AUDIT SUMMARY")
    print("=" * 60)
    print(f"Sample size: {len(sample_ids)}")
    print(f"Pairs with cosine similarity >= {args.cosine_threshold}: {len(rows_out)}")
    print(f"  ...already caught by pHash (Hamming<=3): {len(rows_out) - len(missed_by_phash)}")
    print(f"  ...MISSED by pHash: {len(missed_by_phash)}")

    if missed_by_phash:
        missed_by_phash.sort(key=lambda x: -x[2])
        top = missed_by_phash[:20]
        thumb = 200
        sheet = Image.new("RGB", (2 * thumb, len(top) * thumb), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for row_idx, (id1, id2, sim, phd) in enumerate(top):
            im1 = Image.open(args.images_dir / f"{id1}.jpg").convert("RGB").resize((thumb, thumb))
            im2 = Image.open(args.images_dir / f"{id2}.jpg").convert("RGB").resize((thumb, thumb))
            sheet.paste(im1, (0, row_idx * thumb))
            sheet.paste(im2, (thumb, row_idx * thumb))
            draw.text((2, row_idx * thumb + 2), f"cos={sim:.3f} phash_dist={phd}", fill=(255, 255, 0))
        sheet.save(args.output_dir / "semantic_dedup_missed_pairs.png")
        print(f"Visual sheet of top {len(top)} missed pairs: "
              f"{args.output_dir / 'semantic_dedup_missed_pairs.png'}")

    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
