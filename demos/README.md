# Demo Videos

All tracked output clips (H.264, presentation-ready).

| File | Scene | Pipeline | Notes |
|------|-------|----------|-------|
| `tank1_jetson_cpp.mp4` | Abrams, desert | **Jetson C++ TensorRT** | 93 FPS, FP16 |
| `tank4_jetson_cpp.mp4` | Aerial, multi-vehicle | **Jetson C++ TensorRT** | 93 FPS |
| `tank4_jetson_cpp_smooth.mp4` | Aerial, multi-vehicle | **Jetson C++ (smoothed)** | Kalman + EMA, no jitter - **flagship demo** |
| `tank1_pc_fastmot.mp4` | Abrams, desert | PC Python FastMOT | YOLOv8 + DeepSORT-style |
| `tank2_pc_fastmot.mp4` | Tank | PC Python FastMOT | |
| `tank3_pc_fastmot.mp4` | Tank | PC Python FastMOT | |
| `tank4_pc_fastmot.mp4` | Aerial, multi-vehicle | PC Python FastMOT | |
| `tank4_pc_fastmot_reid.mp4` | Aerial, multi-vehicle | PC FastMOT + **PyTorch ReID** | MobileNetV3 appearance matching |
| `cablecar_pc_fastmot.mp4` | Alpine gondola (drone) | PC Python FastMOT | Cable-car detector |

**Suggested presentation order:**
1. `tank4_pc_fastmot.mp4` - baseline Python tracking
2. `tank4_pc_fastmot_reid.mp4` - with appearance ReID
3. `tank4_jetson_cpp_smooth.mp4` - Jetson C++ at 93 FPS, smooth boxes
4. `cablecar_pc_fastmot.mp4` - generalization to a different domain
