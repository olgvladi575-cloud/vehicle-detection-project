# Annotation Guidelines

How vehicles are labeled in this project: the tool, the rules for drawing boxes, examples, the workflow and where the data is stored. Back to [README](../README.md).

## Contents

1. [Tool: CVAT](#1-tool-cvat)
2. [What to label](#2-what-to-label)
3. [How to draw a box](#3-how-to-draw-a-box)
4. [Examples](#4-examples)
5. [Workflow](#5-workflow)
6. [Where the data is stored](#6-where-the-data-is-stored)
7. [Quality check](#7-quality-check)

---

## 1. Tool: CVAT

Labeling is done in [CVAT](https://github.com/cvat-ai/cvat), run locally in Docker. It supports YOLO import/export, so pre-labels from the models can be loaded and corrected instead of drawing every box from scratch.

Install and start:

```bash
git clone https://github.com/cvat-ai/cvat ~/cvat
cd ~/cvat
docker compose up -d
docker compose ps
```

All containers should be `Up`. Create an admin account:

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open <http://localhost:8080> and log in.

Notes:

- CVAT needs several GB of RAM. If it reports `Required services are not healthy`, raise Docker Desktop memory (Settings > Resources) to 6-8 GB and restart.
- Annotations live in Docker volumes. Stop with `docker compose down`; never add `-v`, it deletes all tasks and annotations.

---

## 2. What to label

One class: **`vehicle`** (id 0), shape **Rectangle**.

| Label as `vehicle`                         | Do not label                                  |
| ------------------------------------------ | --------------------------------------------- |
| Cars, vans, pickups                        | People, cyclists without a motor              |
| Trucks, trailers, buses                    | Shadows of vehicles                           |
| Motorcycles, scooters                      | Road markings, signs, containers, roof units  |
| Moving and parked vehicles                 | Vehicles on billboards or screens             |
| Partly hidden vehicles (trees, bridges)    | Objects you cannot identify as a vehicle      |

If a truck pulls a trailer, draw **one box** around the whole truck + trailer.

---

## 3. How to draw a box

- **Tight.** The box touches the visible edges of the vehicle body; no road around it. With tiny vehicles one or two extra pixels matter a lot.
- **Visible part only.** If a vehicle is partly hidden (tree, bridge, another vehicle), box only what is visible.
- **Cut by the frame edge.** Label it if the visible part is still clearly a vehicle; box only the part inside the frame.
- **No shadow.** The shadow is not part of the box.
- **Dense traffic and parking lots.** Every vehicle gets its own box, even when boxes touch. Never one box over a group.
- **Too small or unclear.** If you cannot say it is a vehicle when zoomed in, do not label it. When in doubt, zoom in and compare with the previous and next frame.
- **Consistency across frames.** A vehicle that is labeled in one frame should be labeled in the neighbouring frames too, unless it left the frame or became fully hidden.

---

## 4. Examples

![Frame with pre-labels](images/screen.png)

A pre-labeled frame from the training set: vehicles are small, often dozens per frame, and some are missed or wrongly boxed by the model. Typical corrections:

| Situation                                  | Action                                 |
| ------------------------------------------ | -------------------------------------- |
| Visible vehicle, no box                    | Add a box                              |
| Box on a shadow, marking or roof unit      | Delete the box                         |
| Box much larger than the vehicle           | Tighten to the vehicle edges           |
| One box covers two vehicles                | Delete it, draw two separate boxes     |
| Two boxes on one vehicle                   | Keep one, delete the duplicate         |

---

## 5. Workflow

Every training set (`train_a`, `train_b`, `train_c`, `train_d`) goes through the same steps. Each set is a separate CVAT task.

### 5.1 Pre-label

```bash
python3 scripts/auto_label.py                   # YOLOv8n  -> data/yolo/labels/,  data/yolo/cvat_zip/<set>_yolo.zip
python3 scripts/auto_label_dino.py --role all   # G. DINO  -> data/gdino/labels/, data/gdino/cvat_zip/<set>_gdino.zip
```

To rebuild the archives without re-running a model:

```bash
python3 scripts/make_cvat_zip.py --role all --source yolo
python3 scripts/make_cvat_zip.py --role all --source gdino
```

Archives are in YOLO 1.1 format (`obj.names`, `obj.data`, `train.txt`, `obj_train_data/`).

### 5.2 Create the task in CVAT

Tasks > `+` > Create a new task:

- Name: the set name, e.g. `train_a`.
- Labels: add `vehicle`, type Rectangle. **Required**, otherwise import fails with `Label 'vehicle' is not registered for this task`.
- My computer: select all `.jpg` from `data/frames/<set>/`. The file count must match the set (58 / 94 / 51 / 73).
- Advanced configuration: sorting method **Natural**, image quality **95** (vehicles are tiny).
- Submit & Open.

### 5.3 Import pre-labels

Actions > Upload annotations > format **YOLO 1.1** > mode **Replace** > choose `data/yolo/cvat_zip/<set>_yolo.zip` or `data/gdino/cvat_zip/<set>_gdino.zip` > OK > confirm.

Check: in the Requests tab the import is Finished with no error; in the job editor, Info shows the same number of boxes as the source `.txt` files (for example `train_a`: 56 from YOLOv8n).

### 5.4 Review and correct

Go frame by frame and apply the rules from sections 2-3: add missed vehicles, delete false boxes, tighten loose boxes. Save often (Ctrl+S). Keep a short log per set (see section 7).

### 5.5 Export from CVAT

When all frames of a set are corrected and saved:

1. Tasks > open the task (e.g. `train_a`) > **Actions > Export task dataset**.
2. Export format: **YOLO 1.1**. Save images: **off** (frames are already in `data/frames/`).
3. OK. When the export is ready (Requests tab), download the zip.
4. Rename it to the set name and put it into `data/cvat_export/`:

   ```
   data/cvat_export/train_a.zip
   data/cvat_export/train_b.zip
   ...
   ```

5. Unpack the labels into the final dataset folder:

   ```bash
   python3 scripts/import_cvat_export.py --role train_a   # or --role all
   ```

   The script copies `obj_train_data/*.txt` to `data/annotations/<set>/`, writes an empty `.txt` for frames without vehicles, checks that every label belongs to a frame of this set and that only class `0` is used, and prints the number of boxes and of frames without boxes. Write that box count into the log (section 7) and into the labeling status table in the README.

Result for each set:

```
data/
├── frames/train_a/frame_0001.jpg ...        images
└── annotations/train_a/frame_0001.txt ...   corrected labels, same file names
```

`data/annotations/` is the dataset used for training. The CVAT export zips in `data/cvat_export/` are not committed (`*.zip` is in `.gitignore`).

---

## 6. Where the data is stored

| Path                           | Content                                       |
| ------------------------------ | --------------------------------------------- |
| `data/raw_videos/<set>/`       | Source videos                                 |
| `data/frames/<set>/`           | Frames `frame_0001.jpg`, ...                  |
| `data/yolo/labels/<set>/`      | YOLOv8n pre-labels, one `.txt` per frame      |
| `data/yolo/cvat_zip/`          | CVAT import archives `<set>_yolo.zip`         |
| `data/gdino/labels/<set>/`     | Grounding DINO pre-labels                     |
| `data/gdino/cvat_zip/`         | CVAT import archives `<set>_gdino.zip`        |
| `data/cvat_export/<set>.zip`   | Export from CVAT after review (YOLO 1.1)      |
| `data/annotations/<set>/`      | **Final labels** after review (used to train) |
| CVAT Docker volumes            | Tasks and annotations in progress             |

Label file format (YOLO): one line per box, `0 x_center y_center width height`, all values normalized to 0-1. The `.txt` file has the same name as its frame.

---

## 7. Quality check

- After each set, record in a short log: boxes added, deleted, tightened, and the final box count.
- Re-open a random sample of corrected frames and check them against the rules above.
- Rules may only be changed based on training frames. The eval clip is never used to tune labeling rules.
