"""
Imports corrected labels from a CVAT export (YOLO 1.1 format) into data/annotations/<role>/.

Put the CVAT export at data/cvat_export/<role>.zip (e.g. data/cvat_export/train_a.zip).
The script:
  - takes obj_train_data/*.txt from the archive (images, if any, are ignored)
  - writes one .txt per frame in data/frames/<role>/ (frame without boxes -> empty .txt)
  - with --frames, writes only the listed frames (e.g. a demo subset that was corrected)
  - removes exact duplicate boxes (same box twice)
  - stops if the archive has no YOLO labels, labels for unknown frames,
    a class other than 0 (vehicle), or if half or more of a frame's boxes are duplicates
  - prints the box count

  python3 scripts/import_cvat_export.py --role train_a
  python3 scripts/import_cvat_export.py --role all
  python3 scripts/import_cvat_export.py --role train_c --frames 1-6
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
FRAMES_DIR = ROOT / "data" / "frames"
EXPORT_DIR = ROOT / "data" / "cvat_export"
OUT_DIR = ROOT / "data" / "annotations"
ROLES = ["train_a", "train_b", "train_c", "train_d"]


def parse_frames(spec: str) -> set[str]:
    """'1-6' or '1,5,10-12' -> {'frame_0001', ...}"""
    nums = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        nums.update(range(int(a), int(b or a) + 1))
    return {f"frame_{n:04d}" for n in nums}


def import_role(role: str, only: set[str] | None = None) -> None:
    src = EXPORT_DIR / f"{role}.zip"
    if not src.exists():
        print(f"⚠️  {role}: {src.relative_to(ROOT)} not found, skipping")
        return

    with zipfile.ZipFile(src) as z:
        labels = {
            Path(n).stem: z.read(n).decode()
            for n in z.namelist()
            if n.startswith("obj_train_data/") and n.endswith(".txt")
        }
    if not labels:
        print(f"❌ {role}: no YOLO labels in {src.name}; export the task as YOLO 1.1, not 'CVAT for images'")
        return

    frames = sorted((FRAMES_DIR / role).glob("frame_*.jpg"))
    extra = set(labels) - {p.stem for p in frames}
    if extra:
        print(f"❌ {role}: export has .txt files with no matching frame: {sorted(extra)[:5]}; check that this is the right archive")
        return
    if only is not None:
        missing = only - {p.stem for p in frames}
        if missing:
            print(f"❌ {role}: frames not in the set: {sorted(missing)}")
            return
        frames = [p for p in frames if p.stem in only]

    out = OUT_DIR / role
    out.mkdir(parents=True, exist_ok=True)
    total, empty, removed = 0, 0, 0
    for p in frames:
        lines = [l for l in labels.get(p.stem, "").splitlines() if l.strip()]
        bad = [l for l in lines if l.split()[0] != "0"]
        if bad:
            print(f"❌ {role}/{p.stem}: class is not 0: {bad[0]!r}; the CVAT task must have only the vehicle label")
            return
        # Exact duplicates (same box twice) are never valid labels: drop them.
        # If half or more of a frame's boxes are duplicates, the whole task was
        # most likely imported twice, so stop instead of guessing.
        unique = list({tuple(l.split()): l for l in lines}.values())
        dup = len(lines) - len(unique)
        if dup and dup * 2 >= len(lines):
            print(f"❌ {role}/{p.stem}: {dup} of {len(lines)} boxes are duplicates; pre-labels were probably imported twice (re-import with mode Replace)")
            return
        (out / f"{p.stem}.txt").write_text("\n".join(unique))
        total += len(unique)
        empty += not unique
        removed += dup

    print(f"{role}: {total} boxes on {len(frames)} frames ({empty} frames without boxes, "
          f"{removed} duplicate boxes removed) -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="Import corrected labels from a CVAT YOLO 1.1 export")
    ap.add_argument("--role", default="all", help="train_a..train_d or all")
    ap.add_argument("--frames", help="import only these frames, e.g. 1-6 or 1,5,10-12 (one role only)")
    args = ap.parse_args()
    if args.frames and args.role == "all":
        ap.error("--frames needs a single --role")
    only = parse_frames(args.frames) if args.frames else None
    for role in (ROLES if args.role == "all" else [args.role]):
        import_role(role, only)


if __name__ == "__main__":
    main()
