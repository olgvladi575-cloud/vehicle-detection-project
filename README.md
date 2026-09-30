# Vehicle detection

![Drone frame with vehicle pre-labels](docs/images/screen.png)

This is a test project. It shows how I plan and carry out the work with data for an object detector: from raw drone video to a labeled dataset, then training and measurement. The focus is on the dataset: every step is documented and can be repeated with the scripts in this repository. Tested on macOS, CPU only, Python 3.13; all commands are run from the project root.

The work follows four stages:

1. [**Plan**](#1-plan): what data is needed and where it comes from.
2. [**Label**](#2-label): pre-labeling with pretrained models, manual correction in CVAT.
3. [**Train**](#3-train): TODO.
4. [**Measure**](#4-measure): TODO.

## Project layout

| Path                                     | Content                                                        |
| ---------------------------------------- | -------------------------------------------------------------- |
| `data/raw_videos/<set>/`                 | Source videos (not in git, download from the source links)     |
| `data/frames/<set>/`                     | Extracted frames (shared by both models)                       |
| `data/yolo_pre_labelling/labels/<set>/`  | Pre-labels from YOLOv8n (YOLO txt)                             |
| `data/yolo_pre_labelling/cvat_zip/`      | CVAT import archives `<set>_yolo.zip` (not in git, generated)  |
| `data/gdino_pre_labelling/labels/<set>/` | Pre-labels from Grounding DINO (YOLO txt)                      |
| `data/gdino_pre_labelling/cvat_zip/`     | CVAT import archives `<set>_gdino.zip` (not in git, generated) |
| `data/cvat_export/<set>.zip`             | Export from CVAT after manual review (YOLO 1.1)                |
| `data/annotations/<set>/`                | Final labels after manual review in CVAT                       |
| `scripts/`                               | Frame extraction, pre-labeling, CVAT import                    |
| `docs/images/`                           | Images for the README                                          |

## 1. Plan

### Data sources

Public videos from Pexels. The training clips were picked to cover different scenes (interchange, rural road, top-down highway, city intersection).

| Set     | Scene                        | Source                                                              |
| ------- | ---------------------------- | ------------------------------------------------------------------- |
| train_a | Highway interchange          | [pexels.com/video/8968356](https://www.pexels.com/video/8968356/)   |
| train_b | Rural highway, light traffic | [pexels.com/video/5382494](https://www.pexels.com/video/5382494/)   |
| train_c | Simple highway, top-down     | [pexels.com/video/8457857](https://www.pexels.com/video/8457857/)   |
| train_d | Urban intersection           | [pexels.com/video/3405804](https://www.pexels.com/video/3405804/)   |
| eval    | City highway, daytime        | [pexels.com/video/32179597](https://www.pexels.com/video/32179597/) |

### Frame extraction

Frames are already in the repository (`data/frames/<set>/`); this section documents how they were produced.

Each video was downloaded from its source link into `data/raw_videos/<set>/` and split into frames with **FFmpeg** ([ffmpeg.org](https://ffmpeg.org), needed only to re-extract frames) at **3 frames per second** (`fps=3`), the same value for all sets.

The script [scripts/extract_frames.sh](scripts/extract_frames.sh) runs this for every set:

```bash
bash scripts/extract_frames.sh
bash scripts/extract_frames.sh train_a
```

For one video it runs:

```bash
ffmpeg -i data/raw_videos/train_a/<video>.mp4 -vf fps=3 data/frames/train_a/frame_%04d.jpg
```

Frames are named `frame_0001.jpg`, `frame_0002.jpg`, ...; each label file later uses the same name (`frame_0001.txt`).

### Extracted frames

| Set       | Frames  |
| --------- | ------- |
| train_a   | 58      |
| train_b   | 94      |
| train_c   | 51      |
| train_d   | 73      |
| **Total** | **276** |
| eval      | TODO    |

## 2. Label

Frames are first pre-labeled by pretrained models, then every frame is reviewed and corrected by hand in **CVAT**.

### Pre-labeling

Two pretrained models were tried, inference only (no fine-tuning), with general weights (COCO / open-vocabulary), not aerial datasets.

Environment (Python 3.10+):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python scripts/auto_label.py                   # YOLOv8n, all sets, conf=0.25
python scripts/auto_label_dino.py --role all   # Grounding DINO tiny, all sets
```

Each script writes one YOLO `.txt` per frame (`0 x_center y_center width height`, normalized) into `data/<model>_pre_labelling/labels/<set>/` and builds a CVAT import archive in `data/<model>_pre_labelling/cvat_zip/`. Weights download on first run (`yolov8n.pt`, about 6 MB, into the project root; `IDEA-Research/grounding-dino-tiny`, about 700 MB, into the Hugging Face cache).

### Pre-labeling comparison (all training sets)

| Set       | Frames  | YOLOv8n boxes | Per frame | Grounding DINO boxes | Per frame | DINO run time |
| --------- | ------- | ------------- | --------- | -------------------- | --------- | ------------- |
| train_a   | 58      | 28            | ~0.5      | 1608                 | ~27.7     | ~5 min        |
| train_b   | 94      | 26            | ~0.3      | 806                  | ~8.6      | ~15 min       |
| train_c   | 51      | 128           | ~2.5      | 1513                 | ~29.7     | ~10 min       |
| train_d   | 73      | 203           | ~2.8      | 2177                 | ~29.8     | ~15 min       |
| **Total** | **276** | **385**       | ~1.4      | **6104**             | ~22.1     |               |

- YOLOv8n misses most vehicles: the input is downscaled and vehicles are only a few pixels wide. Example: `train_a/frame_0032` has one box but dozens of visible vehicles.
- Grounding DINO finds about 16x more boxes overall (57x on train_a). More boxes does not mean better: some are likely false positives (shadows, road markings). A sampled precision check is **TODO**.
- train_b (rural highway, light traffic) gives the fewest boxes for both models, which matches its sparse traffic.
- Run times are approximate (wall clock, noted by hand).
- **Decision: Grounding DINO pre-labels are the starting point for manual correction in CVAT.** YOLOv8n misses almost all vehicles, so correcting it would mean drawing nearly every box by hand. With Grounding DINO most vehicles are already boxed, and removing false positives is faster than adding missed boxes. The decision is based on training data only.

### CVAT setup

Labeling is done in [CVAT](https://github.com/cvat-ai/cvat), run locally in Docker (requires [Docker Desktop](https://www.docker.com/products/docker-desktop/)). It supports YOLO import/export, so pre-labels from the models can be loaded and corrected instead of drawing every box from scratch.

```bash
git clone https://github.com/cvat-ai/cvat ~/cvat
cd ~/cvat
docker compose up -d
docker compose ps                  # all containers should be "Up"
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open <http://localhost:8080> and log in with the created account.

- CVAT needs several GB of RAM. If it reports `Required services are not healthy`, raise Docker Desktop memory (Settings > Resources) to 6-8 GB and restart.
- Annotations live in Docker volumes. Stop with `docker compose down`; never add `-v`, it deletes all tasks and annotations.

### CVAT workflow

Each set (`train_a` ... `train_d`) is a separate CVAT task.

1. **Create the task:** Tasks > `+` > Create a new task.
   - Name: the set name, e.g. `train_a`.
   - Labels: add `vehicle`, type Rectangle. Required, otherwise import fails with `Label 'vehicle' is not registered for this task`.
   - My computer: select all `.jpg` from `data/frames/<set>/`. The count must match the set (58 / 94 / 51 / 73).
   - Advanced configuration: sorting method **Natural**, image quality **95** (vehicles are tiny).
2. **Import pre-labels:** Actions > Upload annotations > **YOLO 1.1** > mode Replace > `data/gdino_pre_labelling/cvat_zip/<set>_gdino.zip`. If the zip is missing, build it: `python scripts/make_cvat_zip.py --role all --source gdino`. Check that the box count in Info matches the source (train_a: 1608).
3. **Correct:** go frame by frame following the [Annotation Guidelines](https://app.notion.com/p/Annotation-Guidelines-6d54e4b0f1dc83f497d181eca8d502ce). Save often (Ctrl+S).
4. **Export:** Actions > Export task dataset > **YOLO 1.1**, Save images off. Download the zip from the Requests tab, rename it to `<set>.zip` and put it into `data/cvat_export/`.
5. **Collect final labels:**

   ```bash
   python scripts/import_cvat_export.py --role train_a                # one set
   python scripts/import_cvat_export.py --role all                    # all sets
   python scripts/import_cvat_export.py --role train_c --frames 1-6   # only the corrected frames
   ```

- **How to annotate** (what counts as a vehicle, how to draw a box, examples, quality check): 📄 [**Annotation Guidelines** (Notion)](https://app.notion.com/p/Annotation-Guidelines-6d54e4b0f1dc83f497d181eca8d502ce).

### Labeling status

A full manual pass over all 276 frames is the next step. To demonstrate the annotation process end to end, the first 6 frames of `train_c` were corrected by hand in CVAT and exported into the dataset:

| Set     | Frames corrected          | Boxes before (DINO) | Boxes after correction | Stored in                   |
| ------- | ------------------------- | ------------------- | ---------------------- | --------------------------- |
| train_c | 6 of 51 (frame_0001-0006) | 89                  | 53                     | `data/annotations/train_c/` |
| train_a | 0 of 58                   | 1608                | TODO                   |                             |
| train_b | 0 of 94                   | 806                 | TODO                   |                             |
| train_d | 0 of 73                   | 2177                | TODO                   |                             |

On these 6 frames the box count went from 89 to 53 (-40%): a large part of the Grounding DINO boxes were wrong or redundant, which confirms that pre-labels cannot be used without manual review.

## 3. Train

**TODO.**

## 4. Measure

**TODO.**
