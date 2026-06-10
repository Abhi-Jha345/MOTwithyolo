#!/bin/bash
# Usage: ./track_cablecar.sh input.mp4 [output.mp4]
set -e

INPUT="$1"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input_video.mp4> [output.mp4]"
    exit 1
fi

BASENAME=$(basename "$INPUT" .mp4)
OUTPUT="${2:-${BASENAME}_tracked.mp4}"
TMP="${OUTPUT%.mp4}_tmp.mp4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../venv/bin/activate"

echo "Tracking cable cars: $INPUT -> $OUTPUT"

python "$SCRIPT_DIR/app.py" \
    -i "$INPUT" \
    -o "$TMP" \
    -c "$SCRIPT_DIR/cfg/mot_cablecar.json" \
    -l "$SCRIPT_DIR/cablecar.names" \
    -m

echo "Re-encoding to H.264..."
ffmpeg -i "$TMP" -vcodec libx264 -crf 23 -preset fast -an "$OUTPUT" -y -loglevel error
rm "$TMP"

echo "Done: $OUTPUT"
