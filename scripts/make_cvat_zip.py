"""
Збирає zip для імпорту в CVAT (формат YOLO 1.1) з кадрів і YOLO-розмітки.

Результат: data/yolo/cvat_zip/train_a_yolo.zip, data/gdino/cvat_zip/train_a_gdino.zip, ...

Викликається автоматично з auto_label.py / auto_label_dino.py,
а також окремо:
  python3 scripts/make_cvat_zip.py --role train_a --source yolo    # YOLOv8n
  python3 scripts/make_cvat_zip.py --role all --source gdino       # Grounding DINO
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # корінь проєкту
FRAMES_DIR = ROOT / "data" / "frames"
ROLES = ["train_a", "train_b", "train_c", "train_d"]

# source = модель; дані кожної моделі лежать у data/<source>/:
#   labels/<role>/*.txt  — розмітка
#   cvat_zip/            — архіви для імпорту в CVAT
SOURCES = ["yolo", "gdino"]

OBJ_DATA = (
    "classes = 1\n"
    "names = data/obj.names\n"
    "train = data/train.txt\n"
    "valid = data/train.txt\n"
    "backup = backup/\n"
)


def build_zip(role: str, source: str = "yolo") -> Path | None:
    """Створює data/<source>/cvat_zip/<role>_<source>.zip. Повертає шлях до архіву або None."""
    source_dir = ROOT / "data" / source
    frames = sorted((FRAMES_DIR / role).glob("frame_*.jpg"))
    labels_dir = source_dir / "labels" / role
    if not frames or not labels_dir.exists():
        print(f"⚠️  {role}: немає кадрів або розмітки ({labels_dir}) — zip не створено")
        return None

    zip_dir = source_dir / "cvat_zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    out = zip_dir / f"{role}_{source}.zip"
    if out.exists():
        out.unlink()  # завжди збираємо з нуля

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "vehicle\n")
        z.writestr("obj.data", OBJ_DATA)
        z.writestr("train.txt", "".join(f"data/obj_train_data/{p.name}\n" for p in frames))
        for p in frames:
            z.write(p, f"obj_train_data/{p.name}")
            txt = labels_dir / f"{p.stem}.txt"
            z.writestr(f"obj_train_data/{p.stem}.txt", txt.read_text() if txt.exists() else "")

    print(f"  zip: {out.relative_to(ROOT)} ({len(frames)} кадрів)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Build CVAT (YOLO 1.1) import zips")
    ap.add_argument("--role", default="all", help="train_a..train_d або all")
    ap.add_argument("--source", choices=SOURCES, default="yolo", help="yolo = YOLOv8n, gdino = Grounding DINO")
    args = ap.parse_args()
    for role in (ROLES if args.role == "all" else [args.role]):
        build_zip(role, args.source)


if __name__ == "__main__":
    main()
