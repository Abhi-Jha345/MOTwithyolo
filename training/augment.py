#!/usr/bin/env python3
"""
Offline data augmentation for the tank YOLO dataset.

Why offline (and not just train.py's online aug)?
  train.py already augments each epoch (mosaic/mixup/HSV/affine), but that only
  varies the few images you have - it never increases their number. With a tiny
  dataset you also want more physical samples on disk so every epoch sees a richer
  pool and validation isn't starved. This script multiplies each image into N
  augmented copies (with correctly transformed labels) BEFORE training, the online
  aug in train.py then stacks on top.

Emphasis on artificial shadows (the dominant nuisance for outdoor vehicles):
  - hard-edged cast shadows (random polygons),
  - soft ambient shadows (blurred polygon masks),
  - directional light/shade gradients,
plus the usual photometric + geometric transforms.

Geometric transforms re-compute every bounding box; photometric ones leave boxes
untouched. Boxes that fall (mostly) outside the frame after a geometric warp are
dropped via a visibility threshold.

Usage:
    # augment the train split in place (originals kept, copies added):
    python augment.py --data dataset/dataset.yaml --copies 8

    # or point at raw dirs:
    python augment.py --images dataset/images/train --labels dataset/labels/train --copies 8

    # shadows only (e.g. to stress-test a model):
    python augment.py --data dataset/dataset.yaml --copies 5 --only shadow

    # write copies to a separate split instead of mixing into train:
    python augment.py --data dataset/dataset.yaml --copies 8 --out-suffix _aug

    # render QA overlays so you can eyeball boxes survive the warps:
    python augment.py --data dataset/dataset.yaml --copies 4 --overlay
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Label IO  (YOLO: class xc yc w h, all normalised)
# --------------------------------------------------------------------------- #
def read_labels(path: Path):
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        c, xc, yc, w, h = ln.split()[:5]
        boxes.append([int(float(c)), float(xc), float(yc), float(w), float(h)])
    return boxes


def write_labels(path: Path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in boxes))


def yolo_to_corners(box, W, H):
    """[c,xc,yc,w,h] (norm) -> (class, 4x2 corner array in px)."""
    c, xc, yc, w, h = box
    bw, bh = w * W, h * H
    x, y = xc * W, yc * H
    x1, y1, x2, y2 = x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return c, corners


def corners_to_yolo(c, corners, W, H, min_vis=0.25, min_px=4):
    """Axis-aligned box from (possibly warped) corners; None if too clipped/small."""
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    full_area = max(1e-6, (x2 - x1) * (y2 - y1))
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(W), x2), min(float(H), y2)
    bw, bh = cx2 - cx1, cy2 - cy1
    if bw < min_px or bh < min_px:
        return None
    if (bw * bh) / full_area < min_vis:          # mostly outside the frame
        return None
    clamp = lambda v: min(1.0, max(0.0, v))      # kill float dust at the edges
    return [c, clamp((cx1 + bw / 2) / W), clamp((cy1 + bh / 2) / H),
            clamp(bw / W), clamp(bh / H)]


# --------------------------------------------------------------------------- #
# Geometric augmentation (affine; transforms image AND boxes)
# --------------------------------------------------------------------------- #
def random_affine(img, boxes, rng,
                  deg=12.0, scale=(0.8, 1.2), translate=0.08, shear=4.0,
                  flip_p=0.5, border=cv2.BORDER_REPLICATE):
    H, W = img.shape[:2]
    angle = rng.uniform(-deg, deg)
    s = rng.uniform(*scale)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, s)

    # shear
    sh = np.tan(np.deg2rad(rng.uniform(-shear, shear)))
    M[0, 1] += sh

    # translation (fraction of side)
    M[0, 2] += rng.uniform(-translate, translate) * W
    M[1, 2] += rng.uniform(-translate, translate) * H

    flip = rng.random() < flip_p
    if flip:
        # horizontal flip composed as x' = W - x, applied after M
        F = np.array([[-1, 0, W], [0, 1, 0]], dtype=np.float32)
        M3 = np.vstack([M, [0, 0, 1]])
        F3 = np.vstack([F, [0, 0, 1]])
        M = (F3 @ M3)[:2]

    out = cv2.warpAffine(img, M, (W, H), borderMode=border)

    new_boxes = []
    for box in boxes:
        c, corners = yolo_to_corners(box, W, H)
        ones = np.ones((4, 1), dtype=np.float32)
        warped = (np.hstack([corners, ones]) @ M.T)[:, :2]
        nb = corners_to_yolo(c, warped, W, H)
        if nb is not None:
            new_boxes.append(nb)
    return out, new_boxes


# --------------------------------------------------------------------------- #
# Synthetic SHADOWS  (photometric; boxes unchanged)
# --------------------------------------------------------------------------- #
def _random_polygon(W, H, rng, n=(3, 6)):
    k = rng.integers(n[0], n[1] + 1)
    # bias polygon to occupy a sub-region of the frame
    cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H
    rx, ry = rng.uniform(0.2, 0.6) * W, rng.uniform(0.2, 0.6) * H
    angs = np.sort(rng.uniform(0, 2 * np.pi, size=k))
    pts = np.stack([cx + rx * np.cos(angs) * rng.uniform(0.5, 1.0, k),
                    cy + ry * np.sin(angs) * rng.uniform(0.5, 1.0, k)], axis=1)
    return pts.astype(np.int32)


def add_cast_shadow(img, rng, n_shadows=(1, 2), darkness=(0.45, 0.7), soft=True):
    """Darken random polygonal regions to mimic cast shadows. Boxes unaffected."""
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    for _ in range(int(rng.integers(n_shadows[0], n_shadows[1] + 1))):
        poly = _random_polygon(W, H, rng)
        cv2.fillPoly(mask, [poly], 1.0)
    if soft:
        k = int(max(5, (min(H, W) * 0.04) // 2 * 2 + 1))  # odd kernel
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    factor = rng.uniform(*darkness)
    # multiplier map: 1.0 outside shadow, `factor` fully inside, smooth between
    mult = (1.0 - mask * (1.0 - factor))[:, :, None]
    out = np.clip(img.astype(np.float32) * mult, 0, 255).astype(np.uint8)
    return out


def add_light_gradient(img, rng, strength=(0.25, 0.55)):
    """Directional shade gradient across the frame (low sun / vignette-like)."""
    H, W = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    xs = np.linspace(-1, 1, W)[None, :]
    ys = np.linspace(-1, 1, H)[:, None]
    grad = xs * np.cos(ang) + ys * np.sin(ang)          # -1..1 across frame
    grad = (grad - grad.min()) / (grad.ptp() + 1e-6)    # 0..1
    s = rng.uniform(*strength)
    mult = (1.0 - s * grad)[:, :, None]                 # darker toward one side
    return np.clip(img.astype(np.float32) * mult, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Other photometric augmentation (boxes unchanged)
# --------------------------------------------------------------------------- #
def hsv_jitter(img, rng, h=0.015, s=0.5, v=0.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-h, h) * 180) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-s, s)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1 + rng.uniform(-v, v)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def brightness_contrast(img, rng, b=0.25, c=0.25):
    alpha = 1 + rng.uniform(-c, c)   # contrast
    beta = rng.uniform(-b, b) * 255  # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def gaussian_noise(img, rng, sigma=(4, 16)):
    s = rng.uniform(*sigma)
    noise = rng.normal(0, s, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def blur(img, rng, kinds=("gauss", "motion")):
    kind = rng.choice(kinds)
    if kind == "gauss":
        k = int(rng.choice([3, 5, 7]))
        return cv2.GaussianBlur(img, (k, k), 0)
    # motion blur
    k = int(rng.choice([7, 9, 11, 13]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    if rng.random() < 0.5:
        kernel = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1.0)
        base = np.zeros((k, k), np.float32); base[k // 2, :] = 1.0 / k
        kernel = cv2.warpAffine(base, kernel, (k, k))
    return cv2.filter2D(img, -1, kernel)


def fog(img, rng, strength=(0.2, 0.5)):
    H, W = img.shape[:2]
    haze = np.full_like(img, 255)
    a = rng.uniform(*strength)
    return cv2.addWeighted(img, 1 - a, haze, a, 0)


# --------------------------------------------------------------------------- #
# Augmentation pipeline for one output copy
# --------------------------------------------------------------------------- #
PHOTOMETRIC = {
    "shadow": lambda im, b, r: (add_cast_shadow(im, r), b),
    "gradient": lambda im, b, r: (add_light_gradient(im, r), b),
    "hsv": lambda im, b, r: (hsv_jitter(im, r), b),
    "bc": lambda im, b, r: (brightness_contrast(im, r), b),
    "noise": lambda im, b, r: (gaussian_noise(im, r), b),
    "blur": lambda im, b, r: (blur(im, r), b),
    "fog": lambda im, b, r: (fog(im, r), b),
}


def augment_once(img, boxes, rng, only=None, geom_p=0.7):
    """Produce one augmented (img, boxes). `only` restricts to one named op."""
    if only:
        if only == "affine":
            return random_affine(img, boxes, rng)
        op = PHOTOMETRIC.get(only)
        if op is None:
            raise SystemExit(f"--only must be one of: affine,{','.join(PHOTOMETRIC)}")
        return op(img, boxes, rng)

    # 1) optional geometric warp (transforms boxes)
    if rng.random() < geom_p:
        img, boxes = random_affine(img, boxes, rng)

    # 2) always apply a shadow-type op (the whole point), weighted toward shadows
    shade = rng.choice(["shadow", "shadow", "gradient"])
    img, boxes = PHOTOMETRIC[shade](img, boxes, rng)

    # 3) 1-3 extra photometric ops
    pool = ["hsv", "bc", "noise", "blur", "fog"]
    for op in rng.choice(pool, size=int(rng.integers(1, 4)), replace=False):
        img, boxes = PHOTOMETRIC[op](img, boxes, rng)

    return img, boxes


# --------------------------------------------------------------------------- #
# Overlay for QA
# --------------------------------------------------------------------------- #
def overlay(img, boxes):
    vis = img.copy()
    H, W = vis.shape[:2]
    for _, xc, yc, w, h in boxes:
        bw, bh = w * W, h * H
        p1 = (int(xc * W - bw / 2), int(yc * H - bh / 2))
        p2 = (int(xc * W + bw / 2), int(yc * H + bh / 2))
        cv2.rectangle(vis, p1, p2, (0, 0, 255), 2)
    return vis


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def resolve_dirs(args):
    if args.data:
        import re
        txt = Path(args.data).read_text()
        root = Path(args.data).parent
        m = re.search(r"^path:\s*(.+)$", txt, re.M)
        if m:
            root = Path(m.group(1).strip())
        m = re.search(r"^train:\s*(.+)$", txt, re.M)
        train_rel = m.group(1).strip() if m else "images/train"
        images = root / train_rel
        labels = Path(str(images).replace("images", "labels"))
        return images, labels
    if not (args.images and args.labels):
        raise SystemExit("Provide either --data or both --images and --labels.")
    return Path(args.images), Path(args.labels)


def main():
    ap = argparse.ArgumentParser(description="Offline tank-dataset augmentation (shadows + more)")
    ap.add_argument("--data", help="dataset.yaml (uses its train split)")
    ap.add_argument("--images", help="images dir (if not using --data)")
    ap.add_argument("--labels", help="labels dir (if not using --data)")
    ap.add_argument("--copies", type=int, default=8, help="augmented copies per source image")
    ap.add_argument("--only", help="restrict to one op: affine,shadow,gradient,hsv,bc,noise,blur,fog")
    ap.add_argument("--out-suffix", default="", help="write copies to <split><suffix> dirs instead of in place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay", action="store_true", help="also write boxed overlays for QA")
    args = ap.parse_args()

    img_dir, lbl_dir = resolve_dirs(args)
    sources = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS) \
        if img_dir.exists() else []
    if not sources:
        raise SystemExit(f"No source images in {img_dir}. Run sam3_to_yolo.py first.")

    if args.out_suffix:
        out_img = img_dir.parent / (img_dir.name + args.out_suffix)
        out_lbl = lbl_dir.parent / (lbl_dir.name + args.out_suffix)
    else:
        out_img, out_lbl = img_dir, lbl_dir
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    ov_dir = out_img.parent / "aug_overlays"

    rng = np.random.default_rng(args.seed)
    made, dropped_imgs = 0, 0
    for src in sources:
        img = cv2.imread(str(src))
        if img is None:
            continue
        boxes = read_labels(lbl_dir / f"{src.stem}.txt")
        for i in range(args.copies):
            a_img, a_boxes = augment_once(img.copy(), list(boxes), rng, only=args.only)
            if boxes and not a_boxes:
                dropped_imgs += 1          # warp pushed every box out; skip
                continue
            name = f"{src.stem}_aug{i:03d}{src.suffix}"
            cv2.imwrite(str(out_img / name), a_img)
            write_labels(out_lbl / f"{src.stem}_aug{i:03d}.txt", a_boxes)
            if args.overlay:
                ov_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ov_dir / f"{src.stem}_aug{i:03d}.jpg"),
                            overlay(a_img, a_boxes))
            made += 1

    print(f"Source images : {len(sources)}")
    print(f"Copies/image  : {args.copies}  (op={args.only or 'mixed+shadow'})")
    print(f"Augmented made: {made}")
    if dropped_imgs:
        print(f"Skipped       : {dropped_imgs} (all boxes warped out of frame)")
    print(f"Images -> {out_img}")
    print(f"Labels -> {out_lbl}")
    if args.overlay:
        print(f"Overlays -> {ov_dir}  (open to confirm boxes track the warps)")
    if args.out_suffix:
        print("\nNote: copies went to a *separate* dir. To train on them, either point "
              "dataset.yaml's train: at it, or re-run without --out-suffix to mix in.")


if __name__ == "__main__":
    main()
