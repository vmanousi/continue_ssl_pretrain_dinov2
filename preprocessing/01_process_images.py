"""
Process every image in the manifest into the final SSL corpus:

  unlabeled: detect + crop the black-border FOV bounding box
  both:      detect overlay-panel exclusion boxes (NOT removed from pixels --
             recorded as metadata for the DINOv2 crop-sampler to avoid later)
  both:      drop images below a conservative minimum mean-luminance floor
             (the audit showed this catches real junk -- near-black / out-of-
             body / noise frames -- without touching legitimate dim tissue)
  both:      resize so the short side = --short-side (default 256), rescaling
             any recorded panel boxes to match

Writes:
  <output-dir>/images/<image_id>.jpg          the final SSL corpus
  <output-dir>/preprocessing_log.csv          per-image audit trail
  <output-dir>/train.txt                      manifest for the DINOv2 dataset loader
  <output-dir>/exclusion_boxes.csv            image_id -> panel box(es) in FINAL pixel coords

Read-only w.r.t. the raw HyperKvasir source -- only ever reads from there.
"""
import argparse
import csv
from pathlib import Path

import cv2

from common import detect_content_bbox, detect_panel_boxes, mean_luminance

MIN_MEAN_LUMINANCE = 20.0  # conservative floor -- see preprocessing/README or PIPELINE notes
MIN_CONTENT_SIDE = 196  # px, before upscaling to short-side


def process_one(path, source, short_side):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "unreadable"

    crop_box = None
    if source == "unlabeled":
        x, y, w, h = detect_content_bbox(bgr)
        crop_box = (x, y, w, h)
        bgr = bgr[y : y + h, x : x + w]

    h0, w0 = bgr.shape[:2]
    if min(h0, w0) < MIN_CONTENT_SIDE:
        return None, f"too_small_after_crop({w0}x{h0})"

    lum = mean_luminance(bgr)
    if lum < MIN_MEAN_LUMINANCE:
        return None, f"below_luminance_floor({lum:.1f})"

    panel_boxes = detect_panel_boxes(bgr)

    scale = short_side / min(h0, w0)
    new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    panel_boxes_final = [
        (round(x * scale), round(y * scale), round(w * scale), round(h * scale))
        for (x, y, w, h) in panel_boxes
    ]

    return {
        "image": resized,
        "crop_box": crop_box,
        "panel_boxes": panel_boxes_final,
        "final_w": new_w,
        "final_h": new_h,
        "mean_luminance": lum,
    }, "kept"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows (debug)")
    args = parser.parse_args()

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    log_rows = []
    exclusion_rows = []
    kept_ids = []
    drop_reasons = {}

    for i, row in enumerate(rows):
        result, status = process_one(Path(row["path"]), row["source"], args.short_side)

        if status != "kept":
            drop_reasons[status.split("(")[0]] = drop_reasons.get(status.split("(")[0], 0) + 1
            log_rows.append(
                {
                    "image_id": row["image_id"], "source": row["source"], "kept": False,
                    "reason": status, "crop_box": "", "final_w": "", "final_h": "",
                    "mean_luminance": "", "num_panel_boxes": 0,
                }
            )
            continue

        out_path = images_dir / f"{row['image_id']}.jpg"
        cv2.imwrite(str(out_path), result["image"], [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

        log_rows.append(
            {
                "image_id": row["image_id"], "source": row["source"], "kept": True,
                "reason": "kept", "crop_box": result["crop_box"] or "",
                "final_w": result["final_w"], "final_h": result["final_h"],
                "mean_luminance": f"{result['mean_luminance']:.1f}",
                "num_panel_boxes": len(result["panel_boxes"]),
            }
        )
        kept_ids.append(row["image_id"])
        for box in result["panel_boxes"]:
            exclusion_rows.append({"image_id": row["image_id"], "x": box[0], "y": box[1], "w": box[2], "h": box[3]})

        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{len(rows)} ({len(kept_ids)} kept so far)")

    with open(args.output_dir / "preprocessing_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    with open(args.output_dir / "exclusion_boxes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(exclusion_rows)

    with open(args.output_dir / "train.txt", "w") as f:
        for image_id in kept_ids:
            f.write(f"images/{image_id}.jpg\n")

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Input images:  {len(rows)}")
    print(f"Kept:          {len(kept_ids)}")
    print(f"Dropped:       {len(rows) - len(kept_ids)}")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"Images with a detected overlay panel: "
          f"{sum(1 for r in log_rows if r['kept'] and r['num_panel_boxes'] > 0)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
