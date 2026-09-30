"""
Авто-розмітка кадрів Grounding DINO (grounding-dino-tiny), лише inference.

Пише YOLO .txt у data/labels_gdino/<role>/ і, якщо не задано --no-zip,
одразу збирає zip для CVAT: zip/<role>_gdino.zip.

  python3 scripts/auto_label_dino.py --role train_a
  python3 scripts/auto_label_dino.py --role all      # train_a..train_d
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from make_cvat_zip import build_zip

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "car. truck. bus. motorcycle."  # нижній регістр, крапка після кожного слова
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("data/labels_gdino")
ROLES = ["train_a", "train_b", "train_c", "train_d"]


def label_role(processor, model, device, role, box_thr, text_thr):
    out = OUT_DIR / role
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((FRAMES_DIR / role).glob("*.jpg"))
    total = 0

    for p in frames:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        inputs = processor(images=img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        kwargs = dict(text_threshold=text_thr, target_sizes=[(h, w)])
        try:
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_thr, **kwargs)[0]
        except TypeError:  # у новіших версіях transformers параметр називається box_threshold
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_thr, **kwargs)[0]

        lines = []
        for (x1, y1, x2, y2) in res["boxes"].tolist():
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / f"{p.stem}.txt").write_text("\n".join(lines))
        total += len(lines)

    print(f"{role}: {total} боксів на {len(frames)} кадрах")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="train_a", help="train_a..train_d або all")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--no-zip", action="store_true", help="Не створювати zip для CVAT")
    args = ap.parse_args()

    device = "cpu"  # найнадійніше на Mac; mps може давати помилки
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    for role in (ROLES if args.role == "all" else [args.role]):
        label_role(processor, model, device, role, args.box_thr, args.text_thr)
        if not args.no_zip:
            build_zip(role, "gdino")


if __name__ == "__main__":
    main()
