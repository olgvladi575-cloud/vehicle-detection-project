# Vehicle detection from drone footage

![Drone frame with vehicle pre-labels](docs/images/screen.png)

One class: `vehicle` (car, truck, bus, motorcycle merged into a single class, id 0).
Data, pre-labeling and the labeling pipeline are described below.

## Data

### How the frames were obtained

1. The videos were downloaded from public sources by link.
   Source URLs:
   train A, highway interchange, pexels.com/video/8968356
   train B, rural highway, light traffic, pexels.com/video/5382494
   train C, simple highway, top-down, pexels.com/video/8457857
   train D, urban intersection, pexels.com/video/3405804
   eval, city highway, daytime, pexels.com/video/32179597
2. Each video was split into frames with **FFmpeg** (command-line tool, https://ffmpeg.org).

Install FFmpeg (macOS):

```bash
brew install ffmpeg
ffmpeg -version
```

Extract frames, numbered from `frame_0001.jpg`:

```bash
mkdir -p data/frames/train_a
ffmpeg -i <VIDEO_FILE>.mp4 -vf fps=<FPS> -q:v 2 -start_number 1 data/frames/train_a/frame_%04d.jpg
```

`<FPS>` is the sampling rate (frames per second of video kept), `-q:v 2` gives high JPEG quality. Repeat for each clip with its own output folder (`train_b`, `train_c`, `train_d`). Exact `fps` values used: **TODO, fill in.**

### What the data looks like

Four training clips, extracted as `.jpg` frames, all shot from a drone at altitude. Vehicles are very small (a typical box is about 3% of frame width).

| Set       | Frames  |
| --------- | ------- |
| train_a   | 58      |
| train_b   | 94      |
| train_c   | 51      |
| train_d   | 73      |
| **Total** | **276** |

Layout:

```
data/raw_videos/train_a..train_d source videos (local only, not in git)
data/frames/train_a..train_d     frames
data/labels_raw/                 pre-labels from YOLOv8n (YOLO txt format)
data/labels_gdino/               pre-labels from Grounding DINO (side experiment)
data/labels_corrected/           corrected labels (final, after CVAT review)
scripts/auto_label.py            YOLOv8n pre-labeling (+ builds CVAT zip)
scripts/auto_label_dino.py       Grounding DINO pre-labeling (+ builds CVAT zip)
scripts/make_cvat_zip.py         builds CVAT import zips (also usable standalone)
zip/                             generated CVAT import archives (local, not in git)
```

Eval clip: **TODO** (not added yet; it will be kept out of every decision).

## Setup

Requires Python 3, Docker Desktop and FFmpeg.

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Two pretrained models were installed and tried one after the other. Neither is trained or fine-tuned (inference only), and both use general pretrained weights (COCO / open-vocabulary), not aerial datasets.

**Step 1: YOLOv8n (COCO), the first model.** Installed on its own with ultralytics:

```bash
pip install ultralytics
```

The weights `yolov8n.pt` download automatically on the first run of `scripts/auto_label.py`. This model was run on all four sets to produce the initial pre-labels in `data/labels_raw/`. It found very few vehicles (see Results), so a second model was tested.

**Step 2: Grounding DINO (tiny), the second model.** Installed afterwards, as an extra experiment (first on `train_a`, then run on `train_b`, `train_c`, `train_d` as well):

```bash
pip install transformers pillow torch
```

The weights `IDEA-Research/grounding-dino-tiny` (about 700 MB) download from Hugging Face on the first run of `scripts/auto_label_dino.py`. Output goes to a separate folder, `data/labels_gdino/`, so the YOLO labels are not overwritten.

All dependencies are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### CVAT (local, Docker)

CVAT runs as a set of Docker containers.

1. Start Docker Desktop and wait until the engine is running.
2. Get CVAT and start it:
   ```bash
   git clone https://github.com/cvat-ai/cvat ~/cvat
   cd ~/cvat
   docker compose up -d
   docker compose ps          # all containers should be "Up"
   ```
3. Create an admin account:
   ```bash
   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
   ```
4. Open http://localhost:8080 and log in.

Notes:

- CVAT needs several GB of RAM. If it reports `Required services are not healthy` / low memory, raise Docker Desktop memory (Settings > Resources) to 6-8 GB and restart.
- Data is stored in Docker volumes. Stop with `docker compose down`; never use `-v` (it deletes all tasks and annotations).

## Labels

Labels start from model output and are then corrected by hand, not drawn from scratch.

### 1. Pre-label (the CVAT import zip is built automatically)

```bash
python3 scripts/auto_label.py                       # YOLOv8n, all sets, conf=0.25
python3 scripts/auto_label_dino.py --role all       # Grounding DINO, all sets (or one, e.g. --role train_a)
```

Output is one YOLO `.txt` per frame: `0 x_center y_center width height`, normalized.

Right after each set is labeled, the script also builds the CVAT import archive inside the project, so no separate command is needed:

```
zip/train_a_yolo.zip   zip/train_b_yolo.zip   ...    (YOLOv8n labels)
zip/train_a_gdino.zip  zip/train_b_gdino.zip  ...    (Grounding DINO labels)
```

Each archive is in YOLO 1.1 format (`obj.names`, `obj.data`, `train.txt`, `obj_train_data/` with frames and `.txt`). Use `--no-zip` to skip this. To rebuild archives from existing labels without re-running a model:

```bash
python3 scripts/make_cvat_zip.py --role all --source raw      # YOLOv8n labels
python3 scripts/make_cvat_zip.py --role all --source gdino    # Grounding DINO labels
```

The `zip/` folder is generated locally and is not committed (`*.zip` is in `.gitignore`; the archives contain all frames and are large).

### 2. Load each set into CVAT

Each set is a separate CVAT task. For every set (`train_a`, `train_b`, `train_c`, `train_d`):

**a) Take the archive** `zip/<set>_yolo.zip` (or `zip/<set>_gdino.zip`), created in step 1.

**b) Create the task** in CVAT: Tasks > `+` > Create a new task.

- Name: `train_a` (etc.).
- Labels: add `vehicle`, type Rectangle. **This is required**, otherwise import fails with `Label 'vehicle' is not registered for this task`.
- My computer: select all `.jpg` from `data/frames/train_a`. The file counter must match the frame count of the set (58 / 94 / 51 / 73).
- Advanced configuration: sorting method Natural, image quality 95 (vehicles are tiny).
- Submit & Open.

**c) Import the pre-labels:** Actions > Upload annotations > format **YOLO 1.1** > mode Replace > choose the zip from `zip/` > OK > confirm "Replace annotations".

**d) Check the import:** in the Requests tab the import must be Finished with no red error. In the job editor, the Info button shows annotation statistics; the total must match the number of boxes in the source `.txt` files (for train_a: 56 from YOLOv8n).

### 3. Review and correct labeling

Frame by frame: add missed vehicles, delete false boxes, tighten loose boxes. Save often (Ctrl+S). Keep a short log of boxes added, removed and fixed.

### 4. Export

Actions > Export task dataset > YOLO 1.1 (images off). Unzip and copy `obj_train_data/*.txt` to `data/labels_corrected/<set>/`.

### Pre-labeling comparison (train_a, 58 frames, training data only)

| Model               | Settings                                                            | Boxes | Avg per frame |
| ------------------- | ------------------------------------------------------------------- | ----- | ------------- |
| YOLOv8n (COCO)      | conf=0.25, default input size                                       | 56    | ~1            |
| Grounding DINO tiny | box/text threshold 0.25, prompt `car. truck. bus. motorcycle.`, CPU | 1608  | ~28           |

- YOLOv8n misses most vehicles: the input is downscaled and vehicles are only a few pixels wide. Example: `frame_0032` has one box but dozens of visible vehicles.
- Grounding DINO finds about 28x more boxes. More boxes does not mean better: some are likely false positives (shadows, road markings). A sampled precision check is **TODO**.
- Which model is used as the starting point for correction: **TODO** (decided on training data only).

### Grounding DINO on all sets (CPU, same settings as above)

| Set       | Frames  | Boxes    | Avg per frame | Run time (CPU) |
| --------- | ------- | -------- | ------------- | -------------- |
| train_a   | 58      | 1608     | ~28           | n/a            |
| train_b   | 94      | 806      | ~8.6          | ~15 min        |
| train_c   | 51      | 1513     | ~29.7         | ~10 min        |
| train_d   | 73      | 2177     | ~29.8         | ~15 min        |
| **Total** | **276** | **6104** | ~22           |                |

- Run times are approximate (wall clock, noted by hand: train_c 13:50-14:00, train_d 14:00-14:15).
- train_b (rural highway, light traffic) gives far fewer boxes per frame than the other sets, which is consistent with its sparse traffic.
- Box counts are raw model output, not checked for false positives.


## Training

**TODO.**

## Eval setup

**TODO.**

## Results

### Detector metrics on the eval clip

**TODO.**

## What next

- Decide the pre-labeling model from a sampled check on training frames.
- Pre-label and correct train_b, train_c, train_d in CVAT.
