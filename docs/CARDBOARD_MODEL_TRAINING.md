# Cardboard Box Detector Training Guide (Roboflow + YOLOv8)

This guide adds a **separate training workflow** to your repo, so your current real-time CV app remains unchanged.

## What this adds

- `requirements-training.txt` for training-only dependencies.
- `training/cardboard_box_training.py` to download a Roboflow dataset and train a YOLOv8 detector.

Your existing runtime app (`run_cv.py`, MediaPipe processors, finger counting, box contour detection) is untouched.

---

## 1) Do I need to download the dataset to my laptop?

**Yes — for local training, the dataset must exist locally while training runs.**

You have two options:

1. **Recommended:** let the script download from Roboflow automatically via API key.
2. Manually export/download dataset from Roboflow and point the script at your local `data.yaml`.

No permanent database server is required. You only need the image/label files available on disk during training.

---

## 2) Minimum training conditions

For a smooth experience:

- Python 3.10+ (3.11 works well).
- At least ~8GB RAM (16GB preferred).
- Disk space: at least 5-10GB free (dataset + checkpoints + logs).
- GPU strongly recommended (NVIDIA CUDA) for speed; CPU works but is slower.

If using only CPU, reduce `--batch` (e.g., 4 or 8) and possibly `--imgsz` (e.g., 512).

---

## 3) Copy/paste install commands

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt
```

If you want to keep runtime and training deps fully separate, use a dedicated venv for training.

---

## 4) Get Roboflow API key

1. Log in to Roboflow.
2. Open your account settings/API section.
3. Copy your private API key.

Then set it in shell:

```bash
export ROBOFLOW_API_KEY="YOUR_KEY_HERE"
```

---

## 5) Train the model (auto-download dataset)

Replace `--version` with your dataset version number from Roboflow.

```bash
python training/cardboard_box_training.py \
  --version 1 \
  --workspace dataset-t7hz7 \
  --project cardboard-eupc8 \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

### GPU example

```bash
python training/cardboard_box_training.py \
  --version 1 \
  --workspace dataset-t7hz7 \
  --project cardboard-eupc8 \
  --model yolov8s.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

---

## 6) Train from a dataset already on disk

If you already downloaded/exported the dataset:

```bash
python training/cardboard_box_training.py \
  --skip-download \
  --data-yaml /absolute/path/to/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

---

## 7) Where outputs go

By default, outputs are saved under:

- `cardboard_training/yolov8_cardboard/` (training run)
- `cardboard_training/yolov8_cardboard_test_predictions/` (sample predictions)

Main model artifact to use later:

- `cardboard_training/yolov8_cardboard/weights/best.pt`

---

## 8) How the training script works

`training/cardboard_box_training.py` does this pipeline:

1. Parses CLI args.
2. Downloads Roboflow dataset in YOLOv8 format (unless `--skip-download`).
3. Reads dataset `data.yaml`.
4. Loads a base YOLOv8 checkpoint (`yolov8n.pt` by default).
5. Trains for configured epochs/image size/batch/device.
6. Runs validation metrics.
7. Runs a sample prediction pass on one split (`test` by default).
8. Prints run directory and where `best.pt` is stored.

---

## 9) Practical hyperparameter starting points

- **Fast baseline:** `yolov8n.pt`, 50-100 epochs, `imgsz 640`.
- **Better accuracy:** `yolov8s.pt` or `yolov8m.pt`, 100-200 epochs (needs stronger GPU).
- **Small objects / far boxes:** try `imgsz 768` or `960`.

If overfitting appears (train improves but val degrades):
- Add/clean data.
- Reduce epochs.
- Use stronger augmentation (Ultralytics defaults are already decent).

---

## 10) Integration with your existing app

This training workflow is intentionally separate and non-destructive.
After training, you can integrate `best.pt` into your app in one of two ways:

1. Keep current contour-based box detector as fallback and add a YOLO processor module.
2. Replace only the box detector path while preserving the modular pipeline.

If you want, the next step is I can wire `best.pt` into your current `cv_modular` processors behind a feature flag, so you can toggle between contour-based and learned detection.
