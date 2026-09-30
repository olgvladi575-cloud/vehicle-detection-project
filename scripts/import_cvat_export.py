"""
Забирає виправлену розмітку з експорту CVAT (формат YOLO 1.1) у data/annotations/<role>/.

Експорт з CVAT кладемо як data/cvat_export/<role>.zip (напр. data/cvat_export/train_a.zip).
Скрипт:
  - бере з архіву obj_train_data/*.txt (картинки, якщо є, ігноруються)
  - перевіряє, що на кожен кадр з data/frames/<role>/ є .txt (кадр без боксів -> порожній .txt)
  - перевіряє, що в розмітці лише клас 0 (vehicle)
  - друкує кількість боксів

  python3 scripts/import_cvat_export.py --role train_a
  python3 scripts/import_cvat_export.py --role all
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # корінь проєкту
FRAMES_DIR = ROOT / "data" / "frames"
EXPORT_DIR = ROOT / "data" / "cvat_export"
OUT_DIR = ROOT / "data" / "annotations"
ROLES = ["train_a", "train_b", "train_c", "train_d"]


def import_role(role: str) -> None:
    src = EXPORT_DIR / f"{role}.zip"
    if not src.exists():
        print(f"⚠️  {role}: немає {src.relative_to(ROOT)} — пропускаю")
        return

    with zipfile.ZipFile(src) as z:
        labels = {
            Path(n).stem: z.read(n).decode()
            for n in z.namelist()
            if n.startswith("obj_train_data/") and n.endswith(".txt")
        }

    frames = sorted((FRAMES_DIR / role).glob("frame_*.jpg"))
    extra = set(labels) - {p.stem for p in frames}
    if extra:
        print(f"❌ {role}: у експорті є .txt без кадру: {sorted(extra)[:5]} — перевірте, що це правильний архів")
        return

    out = OUT_DIR / role
    out.mkdir(parents=True, exist_ok=True)
    total, empty = 0, 0
    for p in frames:
        text = labels.get(p.stem, "").strip()
        lines = [l for l in text.splitlines() if l.strip()]
        bad = [l for l in lines if l.split()[0] != "0"]
        if bad:
            print(f"❌ {role}/{p.stem}: клас не 0: {bad[0]!r} — у CVAT має бути лише мітка vehicle")
            return
        (out / f"{p.stem}.txt").write_text("\n".join(lines))
        total += len(lines)
        empty += not lines

    print(f"{role}: {total} боксів на {len(frames)} кадрах ({empty} кадрів без боксів) -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Import corrected labels from a CVAT YOLO 1.1 export")
    ap.add_argument("--role", default="all", help="train_a..train_d або all")
    args = ap.parse_args()
    for role in (ROLES if args.role == "all" else [args.role]):
        import_role(role)


if __name__ == "__main__":
    main()
