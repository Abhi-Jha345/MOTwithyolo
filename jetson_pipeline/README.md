# Jetson C++ TensorRT Tracking Pipeline

A zero-Python-overhead tank tracker for Jetson AGX Orin. Runs at **~93 FPS**
(vs 36 FPS for the Python implementation) - a **2.8x speedup**.

## Components

| File | Role |
|------|------|
| `src/preprocess.cu` | CUDA letterbox + normalize kernel (BGR uint8 -> FP32 CHW) |
| `src/detector.cpp/.hpp` | TensorRT engine wrapper (load, infer, decode) |
| `src/tracker.cpp/.hpp` | Kalman filter + KLT optical flow + Hungarian IoU matching + EMA smoothing |
| `src/main.cpp` | Video I/O loop, drawing, FPS reporting |

## Prerequisites (JetPack 6 / TensorRT 10)

- CUDA 12.x, TensorRT 10.x (`/usr/include/aarch64-linux-gnu/NvInfer.h`)
- OpenCV 4.x with `calib3d`, `video` modules
- CMake >= 3.18

## 1. Export the TensorRT engine

The C++ runtime needs an engine built with the **full** TRT runtime (not the
Python lean runtime). Build it from the ONNX export:

```bash
# On the Jetson, from a trained .pt:
yolo export model=tank_yolo26n.pt format=onnx imgsz=640

# Then build a C++-compatible FP16 engine:
python3 - <<'PY'
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse(open('tank_yolo26n.onnx','rb').read())
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)
open('tank_yolo26n.engine','wb').write(builder.build_serialized_network(network, config))
PY
```

> Engine I/O tensors: `images` (1,3,640,640) FP32 in, `output0` (1,300,6) FP32 out
> - YOLO26 end-to-end format `[x1,y1,x2,y2,conf,cls]`.

## 2. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 3. Run

```bash
./tank_tracker <engine> <input.mp4> <output.mp4> [conf=0.35] [det_skip=1]
```

- `det_skip=1` - detect every frame (smoothest; afforded by ~450 FPS detector)
- `det_skip=5` - detect every 5th, KLT-propagate between (faster, slight jitter)

## Tracker tuning (smoothness)

In `tracker.cpp`:
- **Kalman measurement noise** - high on width/height (`50.0`) heavily damps
  YOLO's box-size pulsing.
- **EMA smoothing** - separate factors: position `0.7`, size `0.9` (size is
  near-constant for a real vehicle).

## Performance (max clocks: `sudo nvpmodel -m 0 && sudo jetson_clocks`)

| Stage | Latency |
|-------|---------|
| CUDA preprocess | <1 ms |
| TRT FP16 inference | ~2 ms |
| Kalman + KLT + draw | ~3 ms |
| **Total per frame** | **~9 ms (93+ FPS)** |
