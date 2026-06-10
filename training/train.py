#!/usr/bin/env python3
"""
Step 2 - Train YOLOv8 on the tank dataset produced by sam3_to_yolo.py.

Usage:
    python train.py --data dataset/dataset.yaml
    python train.py --data dataset/dataset.yaml --model yolov8s.pt --epochs 200 --imgsz 1024

The trained weights land in runs/detect/<name>/weights/best.pt; that path is what
track.py expects via --weights.

Notes on small datasets (you will start with very few images):
  - Heavy augmentation + a small model (yolov8n) reduces overfitting.
  - mosaic/mixup are enabled by default in ultralytics and help a lot here.
  - Validation metrics will be noisy with <10 images; treat early runs as a smoke
    test of the pipeline, not a measure of real accuracy.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 for tank detection")
    ap.add_argument("--data", type=Path, default=Path("dataset/dataset.yaml"))
    ap.add_argument("--model", default="yolov8n.pt",
                    help="base weights: yolov8n/s/m/l/x.pt (n = smallest/fastest)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="-1 lets ultralytics auto-pick")
    ap.add_argument("--device", default=None, help="'0' for GPU 0, 'cpu', or None=auto")
    ap.add_argument("--name", default="tank_yolov8")
    ap.add_argument("--patience", type=int, default=50, help="early-stopping patience")
    args = ap.parse_args()

    from ultralytics import YOLO

    if not args.data.exists():
        raise SystemExit(f"Dataset YAML not found: {args.data}\n"
                         f"Run sam3_to_yolo.py first.")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # augmentation that helps on tiny tank datasets
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        verbose=True,
    )

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print("\nTraining done.")
    print("Best weights:", best.resolve() if best.exists() else f"(see runs/detect/{args.name})")
    print("Next:  python track.py --weights", best, "--source <video.mp4>")


if __name__ == "__main__":
    main()
