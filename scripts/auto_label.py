"""
Авто-розмітка кадрів готовою (не тренованою нами) моделлю YOLOv8n (COCO-pretrained).

Що робить:
  - Бере кадри з data/frames/<role>/
  - Прогоняє через YOLOv8n (ваги COCO скачаються автоматично при першому запуску)
  - Залишає тільки боксии класів car/truck/bus/motorcycle
  - Перемаплює їх усі в один клас "vehicle" (id = 0)
  - Зберігає розмітку у форматі YOLO .txt в data/labels_raw/<role>/
    (один .txt на кадр, формат рядка: "0 x_center y_center width height", нормалізовано 0..1)

Використання:
  python3 scripts/auto_label.py                 # усі ролі train_a..train_d
  python3 scripts/auto_label.py --role train_a  # тільки одна роль (smoke-test)
  python3 scripts/auto_label.py --conf 0.25      # змінити поріг впевненості
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from make_cvat_zip import build_zip

# COCO-класи, які вважаємо "vehicle" для цієї задачі.
VEHICLE_COCO_CLASSES = {"car", "truck", "bus", "motorcycle"}
VEHICLE_CLASS_ID = 0  # єдиний клас у нашій задачі

FRAMES_DIR = Path("data/frames")
LABELS_RAW_DIR = Path("data/labels_raw")

DEFAULT_ROLES = ["train_a", "train_b", "train_c", "train_d"]


def label_role(model: YOLO, role: str, conf: float, make_zip: bool = True) -> None:
    frames_dir = FRAMES_DIR / role
    if not frames_dir.exists():
        print(f"⚠️  {frames_dir} не існує — пропускаю {role}")
        return

    out_dir = LABELS_RAW_DIR / role
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"⚠️  Немає .jpg кадрів у {frames_dir} — пропускаю {role}")
        return

    print(f"→ {role}: розмічаю {len(frame_paths)} кадрів (conf>={conf}) ...")

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
            x, y, w, h = box.xywhn[0].tolist()  # вже нормалізовано 0..1
            lines.append(f"{VEHICLE_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path = out_dir / f"{frame_path.stem}.txt"
        label_path.write_text("\n".join(lines))

        if lines:
            frames_with_boxes += 1
        total_boxes += len(lines)

    print(
        f"  готово: {total_boxes} боксів на {len(frame_paths)} кадрах "
        f"({frames_with_boxes} кадрів з хоча б одним боксом)"
    )
    if make_zip:
        build_zip(role, "raw")


def main():
    parser = argparse.ArgumentParser(description="Auto-label frames with YOLOv8n (COCO)")
    parser.add_argument(
        "--role", type=str, default=None,
        help="Обробити тільки одну роль (напр. train_a). Без аргументу — всі train-ролі.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Поріг впевненості детекції (default: 0.25)",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Не створювати zip для CVAT у zip/ після розмітки.",
    )
    args = parser.parse_args()

    print("Завантажую YOLOv8n (COCO-pretrained) — без тренування, лише inference ...")
    model = YOLO("yolov8n.pt")

    roles = [args.role] if args.role else DEFAULT_ROLES

    for role in roles:
        label_role(model, role, args.conf, make_zip=not args.no_zip)

    print("\nГотово. Сира розмітка збережена в data/labels_raw/<role>/")
    print("zip для CVAT: zip/<role>_yolo.zip. Наступний крок: імпорт у CVAT для перевірки й корекції.")


if __name__ == "__main__":
    main()
