#!/usr/bin/env python3
"""
Convert the tank annotation format in dataset/dataset/train/annotations.json into a
YOLOv8 detection dataset (single class: tank).

Source format (verified):
    annotations.json = {
        "<frame_index>": [            # list of object instances in this frame
            [[x, y], [x, y], ...],    # each instance = a polygon (SAM3 mask contour)
            ...
        ],
        ...
    }
    frames live next to it as  frames/frame_<frame_index>.jpg

Each polygon is reduced to its axis-aligned bounding box (min/max of its points),
then written as a YOLO label. The matching test/ split has frames but NO
annotations.json, so it can't be a labelled val set; we split the labelled train
frames into train/val here and leave test/frames for track.py inference.

Output:
    <out>/images/{train,val}, <out>/labels/{train,val}, <out>/dataset.yaml

Usage:
    python video_ann_to_yolo.py \
        --ann dataset/dataset/train/annotations.json \
        --out dataset/yolo --val-frac 0.2 [--overlay]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

CLASS_NAMES = ["tank"]


def polygon_to_bbox(poly):
    """list of [x,y] -> (x1,y1,x2,y2) in pixels, or None if too few points."""
    if not poly or len(poly) < 3:
        return None
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(b, W, H):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((max(0.0, min(x1, W)), max(0.0, min(x2, W))))
    y1, y2 = sorted((max(0.0, min(y1, H)), max(0.0, min(y2, H))))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    clamp = lambda v: min(1.0, max(0.0, v))
    return [clamp((x1 + bw / 2) / W), clamp((y1 + bh / 2) / H), clamp(bw / W), clamp(bh / H)]


def render_overlay(img, polys, lines, out_path):
    import numpy as np
    H, W = img.shape[:2]
    vis = img.copy()
    for poly in polys:                      # green = original polygon
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
    for ln in lines:                        # red = derived bbox
        _, xc, yc, w, h = (float(v) for v in ln.split())
        bw, bh = w * W, h * H
        cv2.rectangle(vis, (int(xc * W - bw / 2), int(yc * H - bh / 2)),
                      (int(xc * W + bw / 2), int(yc * H + bh / 2)), (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("dataset/yolo"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.ann.read_text())
    frames_dir = args.ann.parent / "frames"

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = args.out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out / "overlays"

    records = []          # (frame_path, [yolo_lines], [polys])
    missing = total_boxes = empty_frames = 0
    for fidx, objects in data.items():
        fpath = frames_dir / f"frame_{fidx}.jpg"
        img = cv2.imread(str(fpath)) if fpath.exists() else None
        if img is None:
            missing += 1
            continue
        H, W = img.shape[:2]
        lines, polys = [], []
        for poly in objects:
            bb = polygon_to_bbox(poly)
            if bb is None:
                continue
            yolo = bbox_to_yolo(bb, W, H)
            if yolo is None:
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
            polys.append(poly)
            total_boxes += 1
        if not lines:
            empty_frames += 1
        records.append((fpath, lines, polys))

    if not records:
        raise SystemExit("No frames with images found - check --ann and frames/ dir.")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, round(len(records) * args.val_frac))
    val_ids = {id(r) for r in records[:n_val]}

    for rec in records:
        fpath, lines, polys = rec
        split = "val" if id(rec) in val_ids else "train"
        shutil.copy2(fpath, args.out / "images" / split / fpath.name)
        (args.out / "labels" / split / f"{fpath.stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "")
        if args.overlay and id(rec) in val_ids:
            render_overlay(cv2.imread(str(fpath)), polys, lines,
                           overlay_dir / f"{fpath.stem}.jpg")

    yaml_path = args.out / "dataset.yaml"
    yaml_path.write_text(
        "# Tank detection dataset (polygons -> bboxes from frame-indexed annotations.json)\n"
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    n_train = len(records) - n_val
    summary = (
        f"frames with images : {len(records)}  (missing/unreadable: {missing})\n"
        f"tank boxes written : {total_boxes}  (frames with no usable box: {empty_frames})\n"
        f"split              : {n_train} train / {n_val} val\n"
        f"dataset.yaml       : {yaml_path}\n"
        + (f"val overlays       : {overlay_dir}\n" if args.overlay else "")
    )
    print(summary)
    # also persist a summary so it survives flaky stdout
    (args.out / "_convert_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
