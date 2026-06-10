# Modifications to FastMOT

This is a fork of [FastMOT](https://github.com/GeekAlexis/FastMOT) adapted for
custom YOLO detectors, modern TensorRT, and headless operation.

## New files / classes

### `fastmot/detector.py` -> `UltralyticsDetector`
Wraps any `ultralytics.YOLO` model (`.pt` or `.engine`) into FastMOT's async
`Detector` ABC. Returns `DET_DTYPE` record arrays. Includes a second-stage
greedy IoU NMS (threshold 0.3) to remove duplicate boxes that survive YOLO's
internal NMS at low confidence.

Config (`cfg/mot*.json`):
```json
"detector_type": "ULTRALYTICS",
"ultralytics_detector_cfg": {
    "weights": "path/to/best.pt",
    "conf_thresh": 0.35, "iou_thresh": 0.25, "imgsz": 640
}
```

### `fastmot/feature_extractor.py` -> `TorchFeatureExtractor`, `NullFeatureExtractor`
The original OSNet ReID uses the TensorRT-7 API (`builder.max_batch_size`), broken
on TensorRT 8+. Replacements:
- **`TorchFeatureExtractor`** - pretrained MobileNetV3-Small backbone (torchvision),
  576-dim L2-normalized embeddings, cosine distance.
- **`NullFeatureExtractor`** - IoU-only fallback (identical unit embeddings).

`mot.py` tries them in order: TRT OSNet -> PyTorch MobileNetV3 -> IoU-only.
Set `"reid_enabled": false` in `mot_cfg` to force IoU-only (fastest).

## Compatibility fixes

| File | Fix |
|------|-----|
| `fastmot/videoio.py` | `WITH_GSTREAMER` now auto-detected from `cv2.getBuildInformation()`; H.264 `avc1` falls back to `mp4v` when no HW encoder |
| `fastmot/mot.py` | `reid_enabled` flag; graceful feature-extractor fallback chain |
| `fastmot/detector.py` | TRT-10-compatible engine loading via ultralytics |

## Environment notes

- `numba 0.65+` requires `coverage >= 7.0` (older versions lack `coverage.types`).
- `transformers == 4.57.1` for the annotation step (5.x breaks LocateAnything).
