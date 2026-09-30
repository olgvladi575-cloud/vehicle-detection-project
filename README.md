# Vehicle detection from drone footage

One class: `vehicle` (car, truck, bus, motorcycle merged into a single class, id 0).
Data, pre-labeling and the labeling pipeline are described below.

## Data

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
data/frames/train_a..train_d     frames
data/labels_raw/                 pre-labels from YOLOv8n (YOLO txt format)
data/labels_gdino/               pre-labels from Grounding DINO (side experiment)
data/labels_corrected/           corrected labels (final, after CVAT review)
scripts/auto_label.py            YOLOv8n pre-labeling
scripts/auto_label_dino.py       Grounding DINO pre-labeling
```

### How the frames were obtained

1. The videos were downloaded from public sources by link (no ready-made aerial datasets are used).
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

`<FPS>` is the sampling rate (frames per second of video kept), `-q:v 2` gives high JPEG quality. Repeat for each clip with its own output folder (`train_b`, `train_c`, `train_d`). Source URLs and the exact `fps` values used: **TODO, fill in.**

Eval clip: **TODO** (not added yet; it will be kept out of every decision).

## Setup

Requires Python 3, Docker Desktop and FFmpeg.

### Python environment and models

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics transformers torch pillow
```

Model weights download automatically on first run: YOLOv8n (`yolov8n.pt`, COCO) via ultralytics, and `IDEA-Research/grounding-dino-tiny` (about 700 MB) via Hugging Face transformers. No model is trained or fine-tuned for pre-labeling.

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

### 1. Pre-label

```bash
python3 scripts/auto_label.py                       # all sets, YOLOv8n, conf=0.25
python3 scripts/auto_label_dino.py --role train_a   # Grounding DINO, one set
```

Output is one YOLO `.txt` per frame: `0 x_center y_center width height`, normalized.

### 2. Load each set into CVAT

Each set is a separate CVAT task. For every set (`train_a`, `train_b`, `train_c`, `train_d`):

**a) Build the import archive** (YOLO 1.1 format), from the project root:

```bash
x=a   # a, b, c or d
W=/tmp/yolo_$x
rm -rf $W && mkdir -p $W/obj_train_data
cp data/labels_raw/train_$x/frame_*.txt $W/obj_train_data/
cp data/frames/train_$x/frame_*.jpg $W/obj_train_data/
echo "vehicle" > $W/obj.names
printf "classes = 1\nnames = data/obj.names\ntrain = data/train.txt\nvalid = data/train.txt\nbackup = backup/\n" > $W/obj.data
(cd $W && ls obj_train_data/*.jpg | sed 's|^|data/|' > train.txt && zip -rq ~/train_${x}_yolo.zip obj.names obj.data train.txt obj_train_data)
```

For Grounding DINO labels use `data/labels_gdino/` instead and name the archive `train_${x}_gdino.zip`.

**b) Create the task** in CVAT: Tasks > `+` > Create a new task.

- Name: `train_a` (etc.).
- Labels: add `vehicle`, type Rectangle. **This is required**, otherwise import fails with `Label 'vehicle' is not registered for this task`.
- My computer: select all `.jpg` from `data/frames/train_a`. The file counter must match the frame count of the set (58 / 94 / 51 / 73).
- Advanced configuration: sorting method Natural, image quality 95 (vehicles are tiny).
- Submit & Open.

**c) Import the pre-labels:** Actions > Upload annotations > format **YOLO 1.1** > mode Replace > choose the zip > OK > confirm "Replace annotations".

**d) Check the import:** in the Requests tab the import must be Finished with no red error. In the job editor, the Info button shows annotation statistics; the total must match the number of boxes in the source `.txt` files (for train_a: 56 from YOLOv8n).

### 3. Review and correct

Frame by frame: add missed vehicles, delete false boxes, tighten loose boxes. Save often (Ctrl+S). Keep a short log of boxes added, removed and fixed.

### 4. Export

Actions > Export task dataset > YOLO 1.1 (images off). Unzip and copy `obj_train_data/*.txt` to `data/labels_corrected/<set>/`.

Correction effort (boxes added / removed / fixed): **TODO**, will be filled in from the review log once the correction is done.

### Pre-labeling comparison (train_a, 58 frames, training data only)

| Model               | Settings                                                            | Boxes | Avg per frame |
| ------------------- | ------------------------------------------------------------------- | ----- | ------------- |
| YOLOv8n (COCO)      | conf=0.25, default input size                                       | 56    | ~1            |
| Grounding DINO tiny | box/text threshold 0.25, prompt `car. truck. bus. motorcycle.`, CPU | 1608  | ~28           |

- YOLOv8n misses most vehicles: the input is downscaled and vehicles are only a few pixels wide. Example: `frame_0032` has one box but dozens of visible vehicles.
- Grounding DINO finds about 28x more boxes. More boxes does not mean better: some are likely false positives (shadows, road markings). A sampled precision check is **TODO**.
- Which model is used as the starting point for correction: **TODO** (decided on training data only).

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
