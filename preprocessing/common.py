"""Shared image-processing helpers for the HyperKvasir preprocessing pipeline."""
import cv2
import numpy as np

DARK_THRESH = 10


def detect_content_bbox(bgr, dark_thresh=DARK_THRESH):
    """Bounding box of the non-black endoscopic field of view (raw unlabeled frames only)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > dark_thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, bgr.shape[1], bgr.shape[0]
    biggest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(biggest)  # x, y, w, h


# The HyperKvasir "ScopeGuide"-style overlay template observed during audit:
# a solid, strongly-saturated teal icon box, always bottom-left, roughly
# consistent relative size; on some frames a second semi-transparent text
# panel (patient ID/name/timestamp) sits top-left. We search only within
# fixed corner windows -- not the whole frame -- because a whole-image color
# search false-positives on bile-stained / greenish mucosa elsewhere in the
# frame (confirmed during audit: a naive whole-image search gave a nonsense
# ~43% "hit rate" on the already-curated labeled subset).
TEAL_LOWER = (75, 100, 60)
TEAL_UPPER = (100, 255, 255)
CORNER_FRAC = 0.35
MIN_ICON_FILL = 0.10  # fraction of the corner search window the detected blob must fill

# The text panel's background isn't a distinct color (semi-transparent dark
# overlay + white/colored text), so it can't be reliably color-detected on
# its own. Both panels come from the same recording template, so when the
# icon box is found, conservatively also exclude a fixed top-left region
# sized from the audit examples -- approximate, meant to be revisited once
# we have more labeled ground truth, not a precise per-image detection.
TEXT_PANEL_REL_SIZE = (0.32, 0.34)  # (width, height) as a fraction of image size


def detect_panel_boxes(bgr):
    """Return a list of (x, y, w, h) exclusion boxes, in this image's own pixel coordinates."""
    h_img, w_img = bgr.shape[:2]
    cw, ch = int(w_img * CORNER_FRAC), int(h_img * CORNER_FRAC)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    bl_region = hsv[h_img - ch : h_img, 0:cw]
    mask = cv2.inRange(bl_region, TEAL_LOWER, TEAL_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) / (cw * ch) < MIN_ICON_FILL:
        return []

    x, y, w, h = cv2.boundingRect(biggest)
    boxes = [(x, h_img - ch + y, w, h)]  # icon box, translated to full-image coords

    # Co-occurring text panel, conservative fixed-size estimate, top-left.
    tw, th = int(w_img * TEXT_PANEL_REL_SIZE[0]), int(h_img * TEXT_PANEL_REL_SIZE[1])
    boxes.append((0, 0, tw, th))

    return boxes


def mean_luminance(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())
