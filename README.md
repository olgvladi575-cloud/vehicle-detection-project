# Vehicle detection from drone footage

![Drone frame with vehicle pre-labels](docs/images/screen.png)

Detect vehicles in drone footage taken at altitude. One class: `vehicle` (id 0). Car, truck, bus and motorcycle are merged, because from this height they are only a few pixels wide and hard to tell apart reliably.

## 1. Plan

### Data sources

Public videos from Pexels. The training clips were picked to cover different scenes (interchange, rural road, top-down highway, city intersection), so the model does not learn only one type of road. The eval clip is a separate video.

| Set     | Scene                        | Source                                                              |
| ------- | ---------------------------- | ------------------------------------------------------------------- |
| train_a | Highway interchange          | [pexels.com/video/8968356](https://www.pexels.com/video/8968356/)   |
| train_b | Rural highway, light traffic | [pexels.com/video/5382494](https://www.pexels.com/video/5382494/)   |
| train_c | Simple highway, top-down     | [pexels.com/video/8457857](https://www.pexels.com/video/8457857/)   |
| train_d | Urban intersection           | [pexels.com/video/3405804](https://www.pexels.com/video/3405804/)   |
| eval    | City highway, daytime        | [pexels.com/video/32179597](https://www.pexels.com/video/32179597/) |

### Split

- **Train:** `train_a`, `train_b`, `train_c`, `train_d`.
- **Eval:** a separate clip, not frames from the training clips. Neighbouring frames of one video are almost identical, so a random frame split would leak into the test. The eval clip is kept out of every decision (model choice, thresholds, labeling rules) until the final measurement.

### Frame extraction

Each video is split into frames with **FFmpeg** ([ffmpeg.org](https://ffmpeg.org)):

```bash
brew install ffmpeg
mkdir -p data/frames/train_a
ffmpeg -i <VIDEO_FILE>.mp4 -vf fps=<FPS> -q:v 2 -start_number 1 data/frames/train_a/frame_%04d.jpg
```

`<FPS>` is how many frames per second of video are kept; `-q:v 2` gives high JPEG quality. Repeat for each clip with its own folder. Exact `fps` values used: **TODO**.

The same for all sets at once (reads `data/raw_videos/<set>/*.mp4`, default `fps=3`):

```bash
./scripts/extract_frames.sh            # all sets
./scripts/extract_frames.sh train_a    # one set
```

### What the data looks like

All frames are `.jpg`, shot from a drone at altitude. Vehicles are very small (a typical box is about 3% of frame width).

| Set       | Frames  |
| --------- | ------- |
| train_a   | 58      |
| train_b   | 94      |
| train_c   | 51      |
| train_d   | 73      |
| **Total** | **276** |
| eval      | TODO    |

### Project layout

Shared data (frames) is kept once; everything produced by a model lives in that model's own folder.

| Path                          | Content                                         | In git              |
| ----------------------------- | ----------------------------------------------- | ------------------- |
| `data/raw_videos/<set>/`      | Source videos                                   | No                  |
| `data/frames/<set>/`          | Extracted frames (shared by both models)        | Yes                 |
| `data/yolo/labels/<set>/`     | Pre-labels from YOLOv8n (YOLO txt)              | Yes                 |
| `data/yolo/cvat_zip/`         | CVAT import archives `<set>_yolo.zip`           | No (generated)      |
| `data/gdino/labels/<set>/`    | Pre-labels from Grounding DINO (YOLO txt)       | Yes                 |
| `data/gdino/cvat_zip/`        | CVAT import archives `<set>_gdino.zip`          | No (generated)      |
| `data/labels_corrected/<set>/`| Final labels after manual review in CVAT        | TODO (after review) |
| `scripts/`                    | Frame extraction, pre-labeling, CVAT zips       | Yes                 |
| `docs/`                       | Annotation guidelines and images                | Yes                 |

---

## 2. Label

Labels are not drawn from scratch. Frames are first pre-labeled by pretrained models, then every frame is reviewed and corrected by hand in **CVAT**. The full process (tool, rules for boxes, examples, where files are stored) is in 📄 [**Annotation Guidelines**](docs/ANNOTATION_GUIDELINES.md).

### Setup

Requires Python 3, Docker Desktop (for CVAT) and FFmpeg.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CVAT installation is described in the [guidelines](docs/ANNOTATION_GUIDELINES.md#1-tool-cvat).

### Pre-labeling

Two pretrained models were tried, inference only (no fine-tuning), with general weights (COCO / open-vocabulary), not aerial datasets.

```bash
python3 scripts/auto_label.py                   # YOLOv8n, all sets, conf=0.25
python3 scripts/auto_label_dino.py --role all   # Grounding DINO tiny, all sets
```

Each script writes one YOLO `.txt` per frame (`0 x_center y_center width height`, normalized) into `data/<model>/labels/<set>/` and builds a CVAT import archive in `data/<model>/cvat_zip/`. Weights download on first run (`yolov8n.pt`; `IDEA-Research/grounding-dino-tiny`, about 700 MB).

### Pre-labeling comparison (train_a, 58 frames)

| Model               | Settings                                                            | Boxes | Avg per frame |
| ------------------- | ------------------------------------------------------------------- | ----- | ------------- |
| YOLOv8n (COCO)      | conf=0.25, default input size                                       | 56    | ~1            |
| Grounding DINO tiny | box/text threshold 0.25, prompt `car. truck. bus. motorcycle.`, CPU | 1608  | ~28           |

- YOLOv8n misses most vehicles: the input is downscaled and vehicles are only a few pixels wide. Example: `frame_0032` has one box but dozens of visible vehicles.
- Grounding DINO finds about 28x more boxes. More boxes does not mean better: some are likely false positives (shadows, road markings). A sampled precision check is **TODO**.
- Which model is used as the starting point for correction: **TODO** (decided on training data only).

### Grounding DINO on all training sets

| Set       | Frames  | Boxes    | Avg per frame | Run time (CPU) |
| --------- | ------- | -------- | ------------- | -------------- |
| train_a   | 58      | 1608     | ~28           | ~5 min         |
| train_b   | 94      | 806      | ~8.6          | ~15 min        |
| train_c   | 51      | 1513     | ~29.7         | ~10 min        |
| train_d   | 73      | 2177     | ~29.8         | ~15 min        |
| **Total** | **276** | **6104** | ~22           |                |

- Run times are approximate (wall clock, noted by hand).
- train_b (rural highway, light traffic) gives far fewer boxes per frame, which matches its sparse traffic.

### Labeling status

| Set     | Pre-labeled | Corrected in CVAT |
| ------- | ----------- | ----------------- |
| train_a | ✅          | TODO              |
| train_b | ✅          | TODO              |
| train_c | ✅          | TODO              |
| train_d | ✅          | TODO              |

---

## 3. Train

**TODO.**

---

## 4. Measure

**TODO.**
