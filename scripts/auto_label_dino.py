"""
Pre-labels frames with Grounding DINO (grounding-dino-tiny), inference only.

Writes YOLO .txt to data/gdino_pre_labelling/labels/<role>/ and, unless --no-zip is given,
builds the CVAT import zip: data/gdino_pre_labelling/cvat_zip/<role>_gdino.zip.

  python scripts/auto_label_dino.py --role train_a
  python scripts/auto_label_dino.py --role all      # train_a..train_d
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from make_cvat_zip import build_zip

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "car. truck. bus. motorcycle."  # lowercase, a period after each word
ROOT = Path(__file__).resolve().parents[1]  # project root
FRAMES_DIR = ROOT / "data" / "frames"
OUT_DIR = ROOT / "data" / "gdino_pre_labelling" / "labels"
ROLES = ["train_a", "train_b", "train_c", "train_d"]


def filter_boxes(boxes, scores, nms_iou):
    """Removes duplicate and group boxes; boxes are (x1, y1, x2, y2) in pixels."""
    if len(boxes) == 0:
        return boxes
    # 1) duplicates: the same vehicle found by several prompts (car + truck).
    #    All classes are written as 0, so NMS ignores the class and keeps the more confident box.
    keep = nms(boxes, scores, iou_threshold=nms_iou)
    boxes = boxes[keep]

    # 2) group boxes: one large box around several vehicles that are also boxed on their own.
    #    A box is dropped if at least 2 smaller boxes lie inside it (>= 90% of their area).
    x1, y1, x2, y2 = boxes.unbind(1)
    area = (x2 - x1) * (y2 - y1)
    iw = (torch.min(x2[:, None], x2[None]) - torch.max(x1[:, None], x1[None])).clamp(min=0)
    ih = (torch.min(y2[:, None], y2[None]) - torch.max(y1[:, None], y1[None])).clamp(min=0)
    inside = (iw * ih) / area[None] >= 0.9  # inside[i, j]: box j lies inside box i
    inside &= area[None] < area[:, None]    # only smaller boxes count
    return boxes[inside.sum(1) < 2]


def label_role(processor, model, device, role, box_thr, text_thr, nms_iou):
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
        except TypeError:  # in newer transformers versions the parameter is called box_threshold
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=box_thr, **kwargs)[0]

        boxes = filter_boxes(res["boxes"], res["scores"], nms_iou)
        lines = []
        for (x1, y1, x2, y2) in boxes.tolist():
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out / f"{p.stem}.txt").write_text("\n".join(lines))
        total += len(lines)

    print(f"{role}: {total} boxes on {len(frames)} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="train_a", help="train_a..train_d or all")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="Boxes overlapping more than this are treated as duplicates")
    ap.add_argument("--no-zip", action="store_true", help="Do not build the CVAT zip")
    args = ap.parse_args()

    device = "cpu"  # most reliable on Mac; mps may fail
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()

    for role in (ROLES if args.role == "all" else [args.role]):
        label_role(processor, model, device, role, args.box_thr, args.text_thr, args.nms_iou)
        if not args.no_zip:
            build_zip(role, "gdino")


if __name__ == "__main__":
    main()
