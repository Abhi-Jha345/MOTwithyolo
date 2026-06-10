#!/usr/bin/env python3
"""
Step 3 - Detect (YOLOv8) -> Track (DeepSORT) -> Depth (Depth Anything v3) on a video.

For each frame:
  1. YOLOv8 detects tanks  -> boxes + confidences
  2. DeepSORT associates boxes across frames -> stable track IDs
  3. Depth Anything v3 produces a dense relative-depth map for the frame
  4. Each track is annotated with its ID and a depth read-out (median depth inside
     its box, mapped to a near/far value), and a colourised depth video is written too.

Usage:
    python track.py --weights runs/detect/tank_yolov8/weights/best.pt \
                    --source path/to/video.mp4 --out out.mp4

    # image folder instead of a video:
    python track.py --weights best.pt --source frames_dir/ --out out.mp4 --fps 15

    # skip depth (faster) :
    python track.py --weights best.pt --source video.mp4 --no-depth

Depth Anything v3 (DAv3) is loaded from Hugging Face via transformers. The default
checkpoint id is configurable with --depth-model; override it if you have a specific
DAv3 checkpoint mirrored locally or under a different repo id.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Frame source: video file, image folder, or webcam index
# --------------------------------------------------------------------------- #
class FrameSource:
    def __init__(self, source: str, fps_hint: float):
        self.kind = None
        self.fps = fps_hint
        p = Path(source)
        if source.isdigit():                       # webcam index
            self.cap = cv2.VideoCapture(int(source))
            self.kind = "cam"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps_hint
        elif p.is_dir():                           # folder of frames
            import re
            def natural_key(f):
                # sort frame_2 before frame_10 (numeric, not lexicographic)
                nums = re.findall(r"\d+", f.stem)
                return (int(nums[-1]) if nums else 0, f.stem)
            self.frames = sorted(
                [f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS],
                key=natural_key)
            if not self.frames:
                raise SystemExit(f"No images in folder {p}")
            self.kind, self.idx = "dir", 0
        else:                                      # video file
            if not p.exists():
                raise SystemExit(f"Source not found: {source}")
            self.cap = cv2.VideoCapture(str(p))
            self.kind = "video"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps_hint

    def read(self):
        if self.kind == "dir":
            if self.idx >= len(self.frames):
                return False, None
            img = cv2.imread(str(self.frames[self.idx]))
            self.idx += 1
            return img is not None, img
        return self.cap.read()

    def release(self):
        if self.kind in ("video", "cam"):
            self.cap.release()


# --------------------------------------------------------------------------- #
# Depth Anything v3 wrapper
# --------------------------------------------------------------------------- #
class DepthAnythingV3:
    """Thin wrapper around the HF transformers depth-estimation pipeline for DAv3."""

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        self.torch = torch
        self.device = device
        print(f"[depth] loading Depth Anything v3: {model_id} on {device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    @property
    def torch_no_grad(self):
        return self.torch.no_grad()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 depth map (H, W); larger = nearer (inverse depth)."""
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch_no_grad:
            pred = self.model(**inputs).predicted_depth  # (1, h, w)
        depth = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        # resize back to the frame size
        depth = cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
        return depth


def colourise_depth(depth: np.ndarray) -> np.ndarray:
    d = depth - depth.min()
    rng = d.max() if d.max() > 1e-6 else 1.0
    d = (d / rng * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="YOLOv8 + DeepSORT + Depth Anything v3")
    ap.add_argument("--weights", required=True, help="trained YOLOv8 weights (best.pt)")
    ap.add_argument("--source", required=True, help="video file, image folder, or cam index")
    ap.add_argument("--out", default="tracked.mp4", help="annotated output video path")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    ap.add_argument("--fps", type=float, default=25.0, help="fps for image-folder sources")
    ap.add_argument("--device", default=None, help="'0','cpu', or None=auto")
    ap.add_argument("--no-depth", action="store_true", help="disable depth estimation")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf",
                    help="HF id for Depth Anything v3 (override with your DAv3 checkpoint)")
    ap.add_argument("--depth-out", default=None, help="optional separate colourised depth video")
    ap.add_argument("--max-age", type=int, default=30, help="DeepSORT frames to keep lost tracks")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="filter to these COCO class IDs (e.g. 2 for car, 0 for person). "
                         "Ignored when using a single-class custom model.")
    ap.add_argument("--label", default=None,
                    help="label prefix shown in the box (default: model class name)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference size in pixels (default 640; use 1280 for 4K to catch small objects)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    detector = YOLO(args.weights)

    # One DeepSort tracker per class - prevents a 'car' detection from ever
    # being matched to a 'boat' track, eliminating cross-class label flips.
    half = torch.cuda.is_available()
    def _make_tracker():
        return DeepSort(max_age=args.max_age, n_init=3, embedder="mobilenet", half=half)

    trackers: dict[str, object] = {}   # cls_name -> DeepSort
    track_labels: dict[str, str] = {}  # "cls:tid" -> cls_name (fixed at confirmation)

    depth = None if args.no_depth else DepthAnythingV3(
        args.depth_model, "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")

    src = FrameSource(args.source, args.fps)

    writer = None
    depth_writer = None
    rng = np.random.default_rng(42)
    colours: dict[str, tuple] = {}

    def colour_for(key: str):
        if key not in colours:
            colours[key] = tuple(int(c) for c in rng.integers(60, 255, size=3))
        return colours[key]

    frame_i = 0
    t0 = time.time()
    while True:
        ok, frame = src.read()
        if not ok:
            break
        frame_i += 1
        H, W = frame.shape[:2]

        # ---- 1. detect - group by class ------------------------------------
        res = detector.predict(frame, conf=args.conf, iou=args.iou,
                               classes=args.classes, imgsz=args.imgsz,
                               device=device, verbose=False)[0]
        dets_by_cls: dict[str, list] = {}
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])
            cls_name = args.label or detector.names.get(cls_id, str(cls_id))
            dets_by_cls.setdefault(cls_name, []).append(
                ([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

        # ---- 2. update one tracker per class --------------------------------
        all_tracks = []  # (cls_name, track)
        for cls_name, dets in dets_by_cls.items():
            if cls_name not in trackers:
                trackers[cls_name] = _make_tracker()
            trk_list = trackers[cls_name].update_tracks(dets, frame=frame)
            all_tracks.extend((cls_name, tr) for tr in trk_list)

        # also tick trackers whose class had no detections this frame so Kalman
        # predictions stay alive and max_age counts down correctly
        for cls_name, trk in trackers.items():
            if cls_name not in dets_by_cls:
                trk.update_tracks([], frame=frame)

        # ---- 3. depth -------------------------------------------------------
        depth_map = depth.infer(frame) if depth is not None else None

        # ---- 4. annotate ----------------------------------------------------
        for cls_name, tr in all_tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            # Lock the label to the class at first confirmation - never flip it
            key = f"{cls_name}:{tid}"
            if key not in track_labels:
                track_labels[key] = cls_name
            fixed_cls = track_labels[key]

            l, t, r, b = (int(v) for v in tr.to_ltrb())
            l, t = max(0, l), max(0, t)
            r, b = min(W - 1, r), min(H - 1, b)
            col = colour_for(key)

            label = f"{fixed_cls} #{tid}"
            if depth_map is not None and r > l and b > t:
                patch = depth_map[t:b, l:r]
                if patch.size:
                    # DAv3 returns inverse depth (bigger = nearer); report a 0..1
                    # "nearness" plus a relative ranking so closer tanks read higher.
                    med = float(np.median(patch))
                    dmin, dmax = float(depth_map.min()), float(depth_map.max())
                    nearness = (med - dmin) / (dmax - dmin + 1e-6)
                    label += f"  near={nearness:.2f}"

            cv2.rectangle(frame, (l, t), (r, b), col, 2)
            cv2.rectangle(frame, (l, t - 22), (l + 11 * len(label), t), col, -1)
            cv2.putText(frame, label, (l + 2, t - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        n_confirmed = sum(1 for _, tr in all_tracks if tr.is_confirmed())
        cv2.putText(frame, f"frame {frame_i} | tracks {n_confirmed}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---- write outputs -------------------------------------------------
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, src.fps, (W, H))
        writer.write(frame)

        if depth_map is not None and args.depth_out:
            dvis = colourise_depth(depth_map)
            if depth_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                depth_writer = cv2.VideoWriter(args.depth_out, fourcc, src.fps, (W, H))
            depth_writer.write(dvis)

        if frame_i % 20 == 0:
            fps = frame_i / (time.time() - t0)
            print(f"  processed {frame_i} frames  ({fps:.1f} fps)")

    src.release()
    if writer:
        writer.release()
    if depth_writer:
        depth_writer.release()
    dt = time.time() - t0
    print(f"\nDone. {frame_i} frames in {dt:.1f}s ({frame_i / dt:.1f} fps).")
    print("Annotated video ->", args.out)
    if args.depth_out:
        print("Depth video     ->", args.depth_out)


if __name__ == "__main__":
    main()
