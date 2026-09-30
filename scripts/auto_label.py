"""
Pre-labels frames with an off-the-shelf (not fine-tuned) YOLOv8n model (COCO-pretrained).

What it does:
  - Reads frames from data/frames/<role>/
  - Runs YOLOv8n (COCO weights download automatically on first run)
  - Keeps only boxes of classes car/truck/bus/motorcycle
  - Maps them all to one class "vehicle" (id = 0)
  - Saves YOLO .txt labels to data/yolo_pre_labelling/labels/<role>/
    (one .txt per frame, line format: "0 x_center y_center width height", normalized 0..1)

Usage:
  python3 scripts/auto_label.py                 # all roles train_a..train_d
  python3 scripts/auto_label.py --role train_a  # one role only (smoke test)
  python3 scripts/auto_label.py --conf 0.25      # change the confidence threshold
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from make_cvat_zip import build_zip

# COCO classes treated as "vehicle" in this task.
VEHICLE_COCO_CLASSES = {"car", "truck", "bus", "motorcycle"}
VEHICLE_CLASS_ID = 0  # the only class in this task

ROOT = Path(__file__).resolve().parents[1]  # project root
FRAMES_DIR = ROOT / "data" / "frames"
LABELS_DIR = ROOT / "data" / "yolo_pre_labelling" / "labels"

DEFAULT_ROLES = ["train_a", "train_b", "train_c", "train_d"]


def label_role(model: YOLO, role: str, conf: float, make_zip: bool = True) -> None:
    frames_dir = FRAMES_DIR / role
    if not frames_dir.exists():
        print(f"⚠️  {frames_dir} does not exist, skipping {role}")
        return

    out_dir = LABELS_DIR / role
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"⚠️  No .jpg frames in {frames_dir}, skipping {role}")
        return

    print(f"→ {role}: labeling {len(frame_paths)} frames (conf>={conf}) ...")

    total_boxes = 0
    frames_with_boxes = 0

    results = model.predict(
        source=[str(p) for p in frame_paths],
        conf=conf,
        verbose=False,
    )

    for frame_path, result in zip(frame_paths, results):
        lines = []
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in VEHICLE_COCO_CLASSES:
                continue
            x, y, w, h = box.xywhn[0].tolist()  # already normalized 0..1
            lines.append(f"{VEHICLE_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path = out_dir / f"{frame_path.stem}.txt"
        label_path.write_text("\n".join(lines))

        if lines:
            frames_with_boxes += 1
        total_boxes += len(lines)

    print(
        f"  done: {total_boxes} boxes on {len(frame_paths)} frames "
        f"({frames_with_boxes} frames with at least one box)"
    )
    if make_zip:
        build_zip(role, "yolo")


def main():
    parser = argparse.ArgumentParser(description="Auto-label frames with YOLOv8n (COCO)")
    parser.add_argument(
        "--role", type=str, default=None,
        help="Process one role only (e.g. train_a). Without it, all train roles.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Do not build the CVAT zip in data/yolo_pre_labelling/cvat_zip/ after labeling.",
    )
    args = parser.parse_args()

    print("Loading YOLOv8n (COCO-pretrained), inference only, no training ...")
    model = YOLO("yolov8n.pt")

    roles = [args.role] if args.role else DEFAULT_ROLES

    for role in roles:
        label_role(model, role, args.conf, make_zip=not args.no_zip)

    print("\nDone. Raw labels saved to data/yolo_pre_labelling/labels/<role>/")
    print("CVAT zip: data/yolo_pre_labelling/cvat_zip/<role>_yolo.zip. Next step: import into CVAT for review and correction.")


if __name__ == "__main__":
    main()
