#!/usr/bin/env python3
"""
Step 1 - Convert SAM3 annotations into a YOLOv8 detection dataset (single class: tank).

SAM3 is text-promptable, so when you run it with the prompt "tank" every returned
instance IS a tank. This script reads the per-image SAM3 JSON files, derives an
axis-aligned bounding box for each instance, and writes YOLO-format label files plus
a dataset.yaml ready for `yolo train`.

It is tolerant of several common SAM3 / COCO export shapes for each annotation:
  - {"bbox": [x, y, w, h], ...}                         (COCO xywh, absolute px)
  - {"bbox": [x1, y1, x2, y2], "bbox_format": "xyxy"}   (explicit xyxy)
  - {"segmentation": {"counts": ..., "size": [h, w]}}   (COCO RLE  -> needs pycocotools)
  - {"segmentation": [[x1,y1,x2,y2,...]]}               (polygon  -> bbox from extremes)

And several top-level container shapes:
  - {"image": {"width", "height"}, "annotations": [...]}
  - {"width", "height", "annotations"/"masks"/"instances": [...]}
  - [ {...}, {...} ]   (a bare list of annotations; image size read from the .jpg)

Usage:
    python sam3_to_yolo.py --src ../sam3_test --out ./dataset --val-frac 0.2
    python sam3_to_yolo.py --src ../sam3_test --out ./dataset --overlay   # QA overlays only

If your SAM3 JSON uses a field this script doesn't recognise, run with --inspect to
print the detected structure of one file, then tell me the field names.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["tank"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Loading & structure detection
# --------------------------------------------------------------------------- #
def find_image_for(json_path: Path, src_dir: Path) -> Path | None:
    """Find the image that goes with an annotation JSON.

    Handles names like `sam3_sa_123_masks.json` -> `sa_123.jpg` as well as the
    simple `sa_123.json` -> `sa_123.jpg` case.
    """
    stem = json_path.stem
    for junk in ("sam3_", "_masks", "_annotated", "_annotations", "_ann"):
        stem = stem.replace(junk, "")
    candidates = [json_path.with_suffix(ext) for ext in IMG_EXTS]
    candidates += [src_dir / f"{stem}{ext}" for ext in IMG_EXTS]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_annotations(data) -> list:
    """Pull the list of per-instance annotation dicts out of any supported container."""
    if isinstance(data, list):
        return data
    for key in ("annotations", "masks", "instances", "objects", "predictions"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def get_image_size(data, image_path: Path | None):
    """Return (width, height), preferring JSON metadata, falling back to the image."""
    if isinstance(data, dict):
        img_meta = data.get("image", data)
        w = img_meta.get("width") or data.get("width")
        h = img_meta.get("height") or data.get("height")
        if w and h:
            return int(w), int(h)
    if image_path is not None and image_path.exists():
        im = cv2.imread(str(image_path))
        if im is not None:
            h, w = im.shape[:2]
            return int(w), int(h)
    raise ValueError("Could not determine image size from JSON or image file.")


# --------------------------------------------------------------------------- #
# Per-annotation bbox extraction
# --------------------------------------------------------------------------- #
def _bbox_from_polygon(seg) -> list[float] | None:
    pts = []
    for poly in seg:
        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
        pts.append(arr)
    if not pts:
        return None
    allpts = np.concatenate(pts, axis=0)
    x1, y1 = allpts.min(axis=0)
    x2, y2 = allpts.max(axis=0)
    return [x1, y1, x2, y2]  # xyxy


def _bbox_from_rle(seg) -> list[float] | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise SystemExit(
            "This SAM3 file uses COCO-RLE segmentation; install pycocotools "
            "(`pip install pycocotools`) or re-export with bbox fields."
        )
    rle = seg
    if isinstance(seg.get("counts"), list):  # uncompressed RLE -> compress
        rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
    x, y, w, h = mask_utils.toBbox(rle).tolist()
    return [x, y, x + w, y + h]  # xyxy


def extract_xyxy(ann: dict) -> list[float] | None:
    """Return an axis-aligned [x1, y1, x2, y2] box in absolute pixels, or None."""
    if "bbox" in ann and ann["bbox"]:
        b = list(map(float, ann["bbox"]))
        fmt = ann.get("bbox_format", "xywh").lower()
        if fmt == "xyxy":
            return b
        x, y, w, h = b  # default COCO xywh
        return [x, y, x + w, y + h]

    seg = ann.get("segmentation")
    if isinstance(seg, dict) and "counts" in seg:
        return _bbox_from_rle(seg)
    if isinstance(seg, list) and seg:
        return _bbox_from_polygon(seg)
    return None


def xyxy_to_yolo(box, img_w, img_h):
    """Absolute xyxy -> normalised YOLO [xc, yc, w, h], clamped. None if degenerate."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
    y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return [(x1 + bw / 2) / img_w, (y1 + bh / 2) / img_h, bw / img_w, bh / img_h]


# --------------------------------------------------------------------------- #
# Conversion driver
# --------------------------------------------------------------------------- #
def convert_file(json_path: Path, src_dir: Path):
    """Return (image_path, img_w, img_h, [yolo_label_lines])."""
    data = json.loads(json_path.read_text())
    image_path = find_image_for(json_path, src_dir)
    img_w, img_h = get_image_size(data, image_path)

    lines = []
    for ann in get_annotations(data):
        if not isinstance(ann, dict):
            continue
        box = extract_xyxy(ann)
        if box is None:
            continue
        yolo = xyxy_to_yolo(box, img_w, img_h)
        if yolo is None:
            continue
        xc, yc, w, h = yolo
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return image_path, img_w, img_h, lines


def render_overlay(image_path: Path, lines, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    H, W = img.shape[:2]
    for ln in lines:
        _, xc, yc, w, h = (float(v) for v in ln.split())
        bw, bh = w * W, h * H
        p1 = (int(xc * W - bw / 2), int(yc * H - bh / 2))
        p2 = (int(xc * W + bw / 2), int(yc * H + bh / 2))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
    cv2.putText(img, f"{len(lines)} tank(s)", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def inspect(src_dir: Path):
    js = sorted(src_dir.glob("*.json"))
    if not js:
        print(f"No .json files in {src_dir}")
        return
    p = js[0]
    data = json.loads(p.read_text())
    print(f"Inspecting: {p.name}")
    print("Top-level type:", type(data).__name__)
    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
    anns = get_annotations(data)
    print(f"Detected {len(anns)} annotation entries.")
    if anns and isinstance(anns[0], dict):
        print("First annotation keys:", list(anns[0].keys()))
        print("First annotation sample:",
              json.dumps({k: anns[0][k] for k in list(anns[0])[:6]}, default=str)[:400])


def build(src_dir: Path, out_dir: Path, val_frac: float, seed: int, overlay_only: bool):
    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json annotation files found in {src_dir}")

    overlay_dir = out_dir / "overlays"
    if not overlay_only:
        for sub in ("images", "labels"):
            if (out_dir / sub).exists():
                shutil.rmtree(out_dir / sub)
        for split in ("train", "val"):
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    records, total_boxes, skipped = [], 0, []
    for jp in json_files:
        try:
            image_path, w, h, lines = convert_file(jp, src_dir)
        except Exception as e:  # keep going; report at the end
            skipped.append(f"{jp.name}: {e}")
            continue
        if image_path is None:
            skipped.append(f"{jp.name}: no matching image found")
            continue
        total_boxes += len(lines)
        records.append((image_path, lines))
        render_overlay(image_path, lines, overlay_dir / f"{image_path.stem}_overlay.jpg")

    if not records:
        raise SystemExit("No convertible (image, annotation) pairs found. "
                         "Run with --inspect to see the JSON structure.")

    print(f"Converted {len(records)} image(s), {total_boxes} tank box(es) total.")
    for s in skipped:
        print("  [skip]", s)
    print(f"QA overlays -> {overlay_dir}  (open these to confirm boxes land on tanks)")

    if overlay_only:
        return

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_frac))) if len(records) > 1 else 0
    val_set = {id(r) for r in records[:n_val]}

    for rec in records:
        image_path, lines = rec
        split = "val" if id(rec) in val_set else "train"
        shutil.copy2(image_path, out_dir / "images" / split / image_path.name)
        (out_dir / "labels" / split / f"{image_path.stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "")

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Tank detection dataset (auto-generated from SAM3 annotations)\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    n_train = len(records) - n_val
    print(f"\nWrote dataset: {n_train} train / {n_val} val")
    print(f"dataset.yaml -> {yaml_path}")
    print("Next:  python train.py --data", yaml_path)


def main():
    ap = argparse.ArgumentParser(description="SAM3 annotations -> YOLOv8 dataset (tank)")
    ap.add_argument("--src", type=Path, required=True, help="dir with SAM3 *.json (+ images)")
    ap.add_argument("--out", type=Path, default=Path("dataset"), help="output dataset dir")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction held out for val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay", action="store_true", help="only render QA overlays")
    ap.add_argument("--inspect", action="store_true", help="print one JSON's structure and exit")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.src)
        return
    build(args.src, args.out, args.val_frac, args.seed, args.overlay)


if __name__ == "__main__":
    main()
