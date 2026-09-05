"""
Read-only visual audit of the HyperKvasir corpus, to measure (not assume) how
prevalent three candidate preprocessing concerns actually are, before deciding
whether to build real preprocessing steps for them:

  1. Black-border / field-of-view geometry on the unlabeled subset (border-crop
     candidate).
  2. The green "ScopeGuide" position-indicator overlay some scopes burn into a
     corner (green-box-removal candidate).
  3. Blur / darkness quality tail (quality-filter candidate).

Does not modify, move, or crop any source image -- only reads, measures, and
writes contact-sheet PNGs + a metrics CSV under --output-dir.

Usage:
    python audit_contact_sheets.py --manifest ../data/manifest_00.csv --output-dir ../data/audit --seed 0
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THUMB = 140  # px, square thumbnail size in contact sheets


def load_bgr(path, max_side=640):
    """Load and, if large, downscale for fast processing (metrics are scale-robust enough for audit purposes)."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return im


def detect_content_bbox(bgr, dark_thresh=10):
    """Bounding box of the non-black endoscopic field of view."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > dark_thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, bgr.shape[1], bgr.shape[0], 1.0
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    content_frac = (w * h) / (bgr.shape[0] * bgr.shape[1])
    return x, y, w, h, content_frac


def detect_green_box(bgr):
    """Look for a compact, strongly-saturated green blob near a corner (ScopeGuide-style overlay)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # OpenCV hue range 0-179; saturated green is roughly 45-85.
    lower = np.array([45, 100, 80])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = bgr.shape[:2]
    img_area = h_img * w_img
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        frac = area / img_area
        if not (0.0008 <= frac <= 0.06):  # small icon, not a large green mucosal region
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        # near a corner: within 30% of width/height of any corner
        near_corner = (
            (cx < 0.3 * w_img or cx > 0.7 * w_img)
            and (cy < 0.3 * h_img or cy > 0.7 * h_img)
        )
        if near_corner:
            candidates.append((x, y, w, h, frac))
    return candidates


def quality_metrics(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_lum = float(gray.mean())
    return laplacian_var, mean_lum


def make_contact_sheet(items, cols, out_path, thumb=THUMB, caption_fn=None):
    """items: list of (PIL.Image already-annotated, caption str or None)."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 14)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (im, caption) in enumerate(items):
        r, c = divmod(i, cols)
        im_resized = im.resize((thumb, thumb))
        x, y = c * thumb, r * (thumb + 14)
        sheet.paste(im_resized, (x, y))
        if caption:
            draw.text((x + 2, y + thumb + 1), caption, fill=(255, 255, 0))
    sheet.save(out_path)


def sample_rows(manifest_path, source, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            if row["source"] == source:
                rows.append(row)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sheet 1: border-crop bbox overlay, unlabeled ----
    print("Sheet 1: border-crop overlay (100 unlabeled)...")
    sample = sample_rows(args.manifest, "unlabeled", 100, args.seed)
    items = []
    for row in sample:
        bgr = load_bgr(row["path"])
        if bgr is None:
            continue
        x, y, w, h, frac = detect_content_bbox(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
        items.append((pil_im, f"{frac:.2f}"))
    make_contact_sheet(items, cols=10, out_path=args.output_dir / "01_border_crop_overlay.png")

    # ---- Sheet 2: green-box detection, unlabeled + labeled ----
    print("Sheet 2: green-box detection (200 unlabeled)...")
    green_hits_unlabeled = 0
    sample = sample_rows(args.manifest, "unlabeled", 200, args.seed)
    items = []
    for row in sample:
        bgr = load_bgr(row["path"])
        if bgr is None:
            continue
        candidates = detect_green_box(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        for (x, y, w, h, frac) in candidates:
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
        if candidates:
            green_hits_unlabeled += 1
        items.append((pil_im, "HIT" if candidates else None))
    make_contact_sheet(items, cols=20, out_path=args.output_dir / "02a_greenbox_unlabeled.png")

    print("Sheet 2b: green-box detection (100 labeled)...")
    green_hits_labeled = 0
    sample = sample_rows(args.manifest, "labeled", 100, args.seed)
    items = []
    for row in sample:
        bgr = load_bgr(row["path"])
        if bgr is None:
            continue
        candidates = detect_green_box(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        for (x, y, w, h, frac) in candidates:
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
        if candidates:
            green_hits_labeled += 1
        items.append((pil_im, "HIT" if candidates else None))
    make_contact_sheet(items, cols=10, out_path=args.output_dir / "02b_greenbox_labeled.png")

    # ---- Sheet 3: quality metrics -- histograms + extremes ----
    print("Sheet 3: quality metrics over 3000 unlabeled + 3000 labeled...")
    metrics_rows = []
    for source, n in (("unlabeled", 3000), ("labeled", 3000)):
        sample = sample_rows(args.manifest, source, n, args.seed)
        for row in sample:
            bgr = load_bgr(row["path"])
            if bgr is None:
                continue
            lap_var, mean_lum = quality_metrics(bgr)
            metrics_rows.append(
                {"image_id": row["image_id"], "path": row["path"], "source": source,
                 "laplacian_var": lap_var, "mean_lum": mean_lum}
            )

    with open(args.output_dir / "quality_metrics_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "path", "source", "laplacian_var", "mean_lum"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, source in enumerate(("unlabeled", "labeled")):
        vals = [r for r in metrics_rows if r["source"] == source]
        lap = [r["laplacian_var"] for r in vals]
        lum = [r["mean_lum"] for r in vals]
        axes[0, col].hist(lap, bins=60, range=(0, np.percentile(lap, 99)))
        axes[0, col].set_title(f"{source}: Laplacian variance (blur)")
        axes[0, col].axvline(np.percentile(lap, 5), color="red", linestyle="--", label="5th pct")
        axes[0, col].legend()
        axes[1, col].hist(lum, bins=60)
        axes[1, col].set_title(f"{source}: mean luminance")
        axes[1, col].axvline(np.percentile(lum, 5), color="red", linestyle="--", label="5th pct")
        axes[1, col].axvline(np.percentile(lum, 95), color="orange", linestyle="--", label="95th pct")
        axes[1, col].legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "03_quality_histograms.png", dpi=130)
    plt.close()

    # Blurriest 100 unlabeled, for visual calibration of a Laplacian threshold.
    unlabeled_sorted = sorted(
        [r for r in metrics_rows if r["source"] == "unlabeled"], key=lambda r: r["laplacian_var"]
    )
    items = []
    for row in unlabeled_sorted[:100]:
        bgr = load_bgr(row["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        items.append((Image.fromarray(rgb), f"{row['laplacian_var']:.0f}"))
    make_contact_sheet(items, cols=10, out_path=args.output_dir / "03b_blurriest_unlabeled.png")

    # Darkest 50 + brightest 50 unlabeled.
    lum_sorted = sorted(
        [r for r in metrics_rows if r["source"] == "unlabeled"], key=lambda r: r["mean_lum"]
    )
    extremes = lum_sorted[:50] + lum_sorted[-50:]
    items = []
    for row in extremes:
        bgr = load_bgr(row["path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        items.append((Image.fromarray(rgb), f"{row['mean_lum']:.0f}"))
    make_contact_sheet(items, cols=10, out_path=args.output_dir / "03c_luminance_extremes_unlabeled.png")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Green-box candidate detections: unlabeled {green_hits_unlabeled}/200 "
          f"({100*green_hits_unlabeled/200:.1f}%), labeled {green_hits_labeled}/100 "
          f"({100*green_hits_labeled/100:.1f}%)")
    print(f"Quality metrics computed on {len(metrics_rows)} sampled images (3000 unlabeled + 3000 labeled target)")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
