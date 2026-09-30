#!/usr/bin/env bash
# Extracts frames from data/raw_videos/<role>/<any name>.mp4
# into data/frames/<role>/frame_0001.jpg, frame_0002.jpg, ...
#
# Usage:
#   ./scripts/extract_frames.sh            # all roles (train_a, train_b, train_c, train_d, eval)
#   ./scripts/extract_frames.sh train_a    # one role only (smoke test)

set -euo pipefail

FPS=3
RAW_DIR="data/raw_videos"
FRAMES_DIR="data/frames"

extract_one() {
  local role_dir="$1"
  local role
  role=$(basename "$role_dir")

  local video_file
  video_file=$(find "$role_dir" -maxdepth 1 -type f -iname "*.mp4" | head -n 1)

  if [ -z "$video_file" ]; then
    echo "⚠️  No .mp4 in $role_dir, skipping"
    return
  fi

  mkdir -p "$FRAMES_DIR/$role"
  echo "→ $role: extracting from $(basename "$video_file") (fps=$FPS) ..."
  ffmpeg -loglevel error -i "$video_file" -vf "fps=$FPS" "$FRAMES_DIR/$role/frame_%04d.jpg"

  local count
  count=$(ls "$FRAMES_DIR/$role" | wc -l | tr -d ' ')
  echo "  done: $count frames"
}

if [ $# -eq 1 ]; then
  # Smoke test on one role
  role_dir="$RAW_DIR/$1"
  if [ ! -d "$role_dir" ]; then
    echo "Folder $role_dir not found. Check the role name."
    exit 1
  fi
  extract_one "$role_dir"
else
  # All roles at once
  for role_dir in "$RAW_DIR"/*/; do
    extract_one "$role_dir"
  done

  echo ""
  echo "Summary:"
  for role_dir in "$RAW_DIR"/*/; do
    role=$(basename "$role_dir")
    count=$(ls "$FRAMES_DIR/$role" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $role: $count frames"
  done
fi
