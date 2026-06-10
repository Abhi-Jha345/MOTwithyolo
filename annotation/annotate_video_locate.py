"""
Video -> YOLOv8 dataset annotator using LocateAnything-3B (NVlabs/Eagle).

Usage:
    python annotate_video_locate.py \
        --video input.mp4 \
        --classes person car truck \
        --out dataset/ \
        [--stride 5]          # process every Nth frame (default: 1)
        [--min-area 0.001]    # drop boxes smaller than this fraction of image (default: 0.001)
        [--model nvidia/LocateAnything-3B]

Output layout (YOLOv8 format):
    dataset/
        images/  *.jpg
        labels/  *.txt   (class_id cx cy w h, normalised 0-1)
        classes.txt
        data.yaml
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

# coordinate helpers

# Regex matching LocateAnything box tokens: <box><x1><y1><x2><y2></box>
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer: str):
    """Return list of (x1,y1,x2,y2) in [0,1000] space from model answer string."""
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in _BOX_RE.findall(answer)]


def box_to_yolo(x1, y1, x2, y2, scale=1000.0):
    """Convert [0,1000] box to YOLO (cx, cy, w, h) normalised to [0,1]."""
    cx = (x1 + x2) / 2.0 / scale
    cy = (y1 + y2) / 2.0 / scale
    w  = (x2 - x1) / scale
    h  = (y2 - y1) / scale
    return cx, cy, w, h


# main

def main():
    parser = argparse.ArgumentParser(
        description="Annotate a video with LocateAnything -> YOLOv8 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    parser.add_argument("--video",    required=True, help="Path to input video")
    parser.add_argument("--classes",  required=True, nargs="+", help="Class names to detect")
    parser.add_argument("--out",      default="dataset", help="Output directory")
    parser.add_argument("--stride",   type=int,   default=1,     help="Process every Nth frame")
    parser.add_argument("--min-area", type=float, default=0.001, help="Drop boxes with w*h < this (0-1)")
    parser.add_argument("--infer-size", type=int, default=1280, help="Max side (px) to resize frame for inference (saves full-res image)")
    parser.add_argument("--model",    default=str(Path(__file__).parent / "weights" / "LocateAnything-3B"))
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    out = Path(args.out)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    (out / "classes.txt").write_text("\n".join(args.classes) + "\n")
    class_idx = {cls: i for i, cls in enumerate(args.classes)}

    # Load model - ~6 GB download from HuggingFace on first run
    print(f"Loading model: {args.model}")
    eagle_path = Path(__file__).parent / "Eagle" / "Embodied"
    if str(eagle_path) not in sys.path:
        sys.path.insert(0, str(eagle_path))

    from locateanything_worker import LocateAnythingWorker  # noqa: E402

    worker = LocateAnythingWorker(args.model)
    print("Model ready.\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    min_area     = args.min_area
    print(f"Video: {total_frames} frames @ {fps:.1f} fps - stride={args.stride}")

    frame_idx   = 0
    saved_count = 0
    total_boxes = 0

    with tqdm(total=(total_frames + args.stride - 1) // args.stride, unit="frame") as pbar:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Downscale for inference only - keeps saved images at original res
            infer_img = pil_img
            if max(pil_img.size) > args.infer_size:
                scale = args.infer_size / max(pil_img.size)
                infer_img = pil_img.resize(
                    (int(pil_img.width * scale), int(pil_img.height * scale)),
                    Image.BILINEAR
                )

            stem    = f"frame_{frame_idx:07d}"
            annotations = []

            # Detect each class separately so box->class mapping is unambiguous
            for cls_name in args.classes:
                cls_id = class_idx[cls_name]
                try:
                    result = worker.detect(infer_img, [cls_name])
                    answer = result["answer"]
                except Exception as e:
                    print(f"\n[warn] frame {frame_idx}, class '{cls_name}': {e}")
                    continue

                for x1, y1, x2, y2 in parse_boxes(answer):
                    cx, cy, w, h = box_to_yolo(x1, y1, x2, y2)
                    if w <= 0 or h <= 0 or w * h < min_area:
                        continue
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w  = max(0.0, min(1.0, w))
                    h  = max(0.0, min(1.0, h))
                    annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            cv2.imwrite(str(img_dir / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(annotations) + ("\n" if annotations else "")
            )

            saved_count += 1
            total_boxes += len(annotations)
            pbar.set_postfix(boxes=len(annotations), total=total_boxes)
            pbar.update(1)
            frame_idx += 1

    cap.release()

    yaml_text = (
        f"path: {out.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(args.classes)}\n"
        f"names: {args.classes}\n"
    )
    (out / "data.yaml").write_text(yaml_text)

    print(f"\nDone. {saved_count} frames, {total_boxes} boxes -> {out}/")
    print(f"  Train with: yolo train data={out}/data.yaml model=yolov8n.pt epochs=50")


if __name__ == "__main__":
    main()
