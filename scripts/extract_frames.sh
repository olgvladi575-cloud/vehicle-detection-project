#!/usr/bin/env bash
# Нарізає кадри з відео у data/raw_videos/<role>/<будь-яка назва>.mp4
# у data/frames/<role>/frame_0001.jpg, frame_0002.jpg, ...
#
# Використання:
#   ./scripts/extract_frames.sh            # усі ролі (train_a, train_b, train_c, train_d, eval)
#   ./scripts/extract_frames.sh train_a    # тільки одна роль (smoke-test)

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
    echo "⚠️  Немає .mp4 у $role_dir — пропускаю"
    return
  fi

  mkdir -p "$FRAMES_DIR/$role"
  echo "→ $role: нарізаю з $(basename "$video_file") (fps=$FPS) ..."
  ffmpeg -loglevel error -i "$video_file" -vf "fps=$FPS" "$FRAMES_DIR/$role/frame_%04d.jpg"

  local count
  count=$(ls "$FRAMES_DIR/$role" | wc -l | tr -d ' ')
  echo "  готово: $count кадрів"
}

if [ $# -eq 1 ]; then
  # Smoke-test на одній ролі
  role_dir="$RAW_DIR/$1"
  if [ ! -d "$role_dir" ]; then
    echo "Папку $role_dir не знайдено. Перевірте назву ролі."
    exit 1
  fi
  extract_one "$role_dir"
else
  # Усі ролі одразу
  for role_dir in "$RAW_DIR"/*/; do
    extract_one "$role_dir"
  done

  echo ""
  echo "Підсумок:"
  for role_dir in "$RAW_DIR"/*/; do
    role=$(basename "$role_dir")
    count=$(ls "$FRAMES_DIR/$role" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $role: $count frames"
  done
fi
