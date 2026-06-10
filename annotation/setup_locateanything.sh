#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_DIR="$SCRIPT_DIR/Eagle"

# Clone Eagle repo
if [ ! -d "$EAGLE_DIR" ]; then
    git clone https://github.com/NVlabs/Eagle.git "$EAGLE_DIR"
    echo "Cloned Eagle to $EAGLE_DIR"
else
    echo "Eagle already cloned at $EAGLE_DIR"
fi

# Activate venv if present, else use system Python
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated venv"
fi

# Install LocateAnything and its dependencies (non-editable; build backend lacks PEP 660)
cd "$EAGLE_DIR/Embodied"
pip install . --quiet

# Extra dependencies for the annotation pipeline
pip install opencv-python-headless tqdm huggingface_hub --quiet

echo ""
echo "Setup complete."
echo "Model will be auto-downloaded from HuggingFace on first run: nvidia/LocateAnything-3B"
echo ""
echo "Usage:"
echo "  python annotate_video_locate.py --video <path> --classes person car truck --out dataset/"
