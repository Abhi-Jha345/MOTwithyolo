# Pre-built TensorRT Engines

> **Platform-specific.** These engines were built for and only run on:
> **Jetson AGX Orin  |  JetPack 6  |  TensorRT 10.3  |  CUDA 12.6  |  sm_87**.
> On any other GPU/TRT version, rebuild from the `.pt`/`.onnx` (see
> [../README.md](../README.md) -> "Export the TensorRT engine").

| Engine | Precision | Runtime | Size | Use with |
|--------|-----------|---------|------|----------|
| `tank_yolo26n_cpp.engine` | FP16 | **full** (C++ compatible) | 8.0 MB | `jetson_pipeline` C++ `tank_tracker` |
| `tank_yolo26n_fp16.engine` | FP16 | lean (Python) | 7.9 MB | Python ultralytics / FastMOT |
| `tank_yolo26n_int8.engine` | INT8 | lean (Python) | 4.8 MB | Python (max speed; end2end NMS disabled on JP6) |

**Important:** the C++ `tank_tracker` requires the **full**-runtime engine
(`*_cpp.engine`). Engines exported by the Python lean runtime
(`*_fp16/int8.engine`) will fail to deserialize in C++ with a
`magicTag` / serialization-version error.

I/O tensors (all): `images` (1,3,640,640) FP32 -> `output0` (1,300,6) FP32,
YOLO26 end-to-end `[x1,y1,x2,y2,conf,cls]`.

## Benchmarks (Jetson AGX Orin, max clocks - detection only)

| Engine | FPS |
|--------|-----|
| FP16 TRT | 133 |
| INT8 TRT | 98 |
| FP32 PyTorch (baseline) | 37 |
