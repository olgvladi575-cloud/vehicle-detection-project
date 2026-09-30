"""
Builds CVAT import zips (YOLO 1.1 format) from frames and YOLO labels.

Output: data/yolo/cvat_zip/train_a_yolo.zip, data/gdino/cvat_zip/train_a_gdino.zip, ...

Called automatically by auto_label.py / auto_label_dino.py,
or standalone:
  python3 scripts/make_cvat_zip.py --role train_a --source yolo    # YOLOv8n
  python3 scripts/make_cvat_zip.py --role all --source gdino       # Grounding DINO
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
FRAMES_DIR = ROOT / "data" / "frames"
ROLES = ["train_a", "train_b", "train_c", "train_d"]

# source = model; each model keeps its data in data/<source>/:
#   labels/<role>/*.txt  - labels
#   cvat_zip/            - CVAT import archives
SOURCES = ["yolo", "gdino"]

OBJ_DATA = (
    "classes = 1\n"
    "names = data/obj.names\n"
    "train = data/train.txt\n"
    "valid = data/train.txt\n"
    "backup = backup/\n"
)


def build_zip(role: str, source: str = "yolo") -> Path | None:
    """Creates data/<source>/cvat_zip/<role>_<source>.zip. Returns the archive path or None."""
    source_dir = ROOT / "data" / source
    frames = sorted((FRAMES_DIR / role).glob("frame_*.jpg"))
    labels_dir = source_dir / "labels" / role
    if not frames or not labels_dir.exists():
        print(f"⚠️  {role}: no frames or labels ({labels_dir}), zip not created")
        return None

    zip_dir = source_dir / "cvat_zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    out = zip_dir / f"{role}_{source}.zip"
    if out.exists():
        out.unlink()  # always rebuild from scratch

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "vehicle\n")
        z.writestr("obj.data", OBJ_DATA)
        z.writestr("train.txt", "".join(f"data/obj_train_data/{p.name}\n" for p in frames))
        for p in frames:
            z.write(p, f"obj_train_data/{p.name}")
            txt = labels_dir / f"{p.stem}.txt"
            z.writestr(f"obj_train_data/{p.stem}.txt", txt.read_text() if txt.exists() else "")

    print(f"  zip: {out.relative_to(ROOT)} ({len(frames)} frames)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build CVAT (YOLO 1.1) import zips")
    ap.add_argument("--role", default="all", help="train_a..train_d or all")
    ap.add_argument("--source", choices=SOURCES, default="yolo", help="yolo = YOLOv8n, gdino = Grounding DINO")
    args = ap.parse_args()
    for role in (ROLES if args.role == "all" else [args.role]):
        build_zip(role, args.source)


if __name__ == "__main__":
    main()
