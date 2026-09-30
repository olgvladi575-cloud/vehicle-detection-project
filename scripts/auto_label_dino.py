import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# нижній регістр, крапка після кожного слова
PROMPT = "car. truck. bus. motorcycle."
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("data/labels_gdino")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="train_a")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.25)
    args = ap.parse_args()

    device = "cpu"  # найнадійніше на Mac; mps може давати помилки
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID).to(device).eval()

    out = OUT_DIR / args.role
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((FRAMES_DIR / args.role).glob("*.jpg"))
    total = 0

    for p in frames:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        inputs = processor(images=img, text=PROMPT,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        kwargs = dict(text_threshold=args.text_thr, target_sizes=[(h, w)])
        try:
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=args.box_thr, **kwargs)[0]
        except TypeError:  # у новіших версіях transformers параметр називається box_threshold
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=args.box_thr, **kwargs)[0]

        lines = []
        for (x1, y1, x2, y2) in res["boxes"].tolist():
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / f"{p.stem}.txt").write_text("\n".join(lines))
        total += len(lines)

    print(f"{args.role}: {total} боксів на {len(frames)} кадрах")


if __name__ == "__main__":
    main()
