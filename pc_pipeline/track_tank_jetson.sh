#!/bin/bash
# Run C++ tank tracker on Jetson and copy result back here.
# Usage: ./track_tank_jetson.sh <input.mp4> [output.mp4]
set -e

INPUT="$1"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input_video.mp4> [output.mp4]"
    exit 1
fi

BASENAME=$(basename "$INPUT" .mp4)
OUTPUT="${2:-${HOME}/Downloads/${BASENAME}_tracked.mp4}"

ENGINE="/home/upbus/line_detection_algorithms/weights/tank_yolo26n_cpp.engine"
TRACKER="/home/upbus/line_detection_algorithms/cpp_pipeline/build/tank_tracker"
REMOTE_IN="/tmp/${BASENAME}.mp4"
REMOTE_OUT="/tmp/${BASENAME}_tracked_tmp.mp4"
REMOTE_H264="/tmp/${BASENAME}_tracked.mp4"

echo "=== Jetson C++ Tank Tracker ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

# 1. Upload video to Jetson
echo "[1/4] Uploading video to Jetson..."
scp -q "$INPUT" "jetson:${REMOTE_IN}"

# 2. Run C++ tracker on Jetson (detect every frame for smooth tracking)
echo "[2/4] Running tracker (YOLO26n FP16 TRT + Kalman, detect every frame)..."
ssh jetson "
    $TRACKER $ENGINE $REMOTE_IN $REMOTE_OUT 0.35 1
"

# 3. Re-encode to H.264 on Jetson 
echo "[3/4] Encoding to H.264..."
ssh jetson "
    ffmpeg -i $REMOTE_OUT \
        -vcodec libx264 -crf 23 -preset fast -an \
        $REMOTE_H264 -y -loglevel error
    rm -f $REMOTE_OUT $REMOTE_IN
"

# 4. Copy result back
echo "[4/4] Copying result back..."
scp -q "jetson:${REMOTE_H264}" "$OUTPUT"
ssh jetson "rm -f $REMOTE_H264"

echo ""
echo "Done: $OUTPUT"
ls -lh "$OUTPUT"
