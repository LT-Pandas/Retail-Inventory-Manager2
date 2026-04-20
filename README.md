# Retail-Inventory-Manager

This repository includes an **absolute-minimum modular computer vision starter** based on OpenCV so you can:

1. Detect how many fingers are currently held up (webcam, real time).
2. Keep a clean module layout so object detection and future CV features are easy to plug in.
3. Detect rectangular boxes at the same time as hands/fingers using a unified processor (no trained box model needed).

The finger-counting processor now uses an OpenCV-only contour/convexity-defect approach so it can run on Raspberry Pi Python 3.13 + Picamera2 setups without MediaPipe.

---

## Project layout

```text
.
├── cv_modular/
│   ├── interfaces.py                  # shared processor contract
│   ├── pipeline.py                    # processor orchestration + webcam loop
│   └── processors/
│       ├── box_detector.py            # contour-based rectangular box detection plugin
│       ├── finger_counter.py          # OpenCV-only hand tracking + finger counting plugin
│       └── object_detector.py         # optional MediaPipe Tasks object detection plugin
├── run_cv.py                          # one command entrypoint
└── requirements.txt
```

---

## Quick start (copy/paste)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you also want optional object detection, install MediaPipe separately (it may not be available on Python 3.13):

```bash
pip install mediapipe
```

### 3) Run finger counting

```bash
python run_cv.py
```

- Press **`q`** to quit.
- You will see hand contour tracking and a `Fingers: N` overlay in the OpenCV window.

---

## Common run options

### Use another camera index

```bash
python run_cv.py --camera-index 1
```

### Tune hand detector compatibility options

```bash
python run_cv.py \
  --min-detection-confidence 0.7 \
  --min-presence-confidence 0.6 \
  --min-tracking-confidence 0.6
```

### If your video feed is **not mirrored**

```bash
python run_cv.py --no-assume-selfie-view
```

### Send live finger count to Arduino over serial (while keeping the webcam view)

```bash
python run_cv.py --serial-port COM5 --serial-baud 115200
```

- Linux/macOS port examples: `/dev/ttyUSB0`, `/dev/ttyACM0`, `/dev/tty.usbmodemXXXX`
- Message format: `FINGERS:<count>\n` (for example `FINGERS:3`)
- Updates are sent only when the count changes, capped at ~20 Hz.

---

### Hand + box detection now run together

```bash
python run_cv.py
```

Optional box tuning:

```bash
python run_cv.py --box-min-area 4000
```

The unified processor overlays both OpenCV hand tracking / `Fingers: N` and rectangular box outlines / `Boxes: N` in the same frame.

---

## MediaPipe object detection integration (for package/box-like objects)

The architecture supports MediaPipe object detection as a second processor.

Run with the built-in default model (auto-downloaded on first run):

```bash
python run_cv.py --detect-objects
```

Use your own model if desired:

```bash
python run_cv.py --detect-objects --object-model /absolute/path/to/model.tflite
```

By default, detections are filtered to box/package-like labels (`box`, `package`, `parcel`, `carton`) so the overlay stays focused on inventory-style objects.

---



## Roboflow people counting (stereo-camera ready)

This project now supports counting people with Roboflow model `crowd-counting-dataset-w3o7w/2`.

1. Export your key:

```bash
export ROBOFLOW_API_KEY=your_key_here
```

2. Run people counting:

```bash
python run_cv.py --count-people
```

For side-by-side stereo streams, choose layout + eye:

```bash
python run_cv.py --count-people --stereo-layout left-right --stereo-eye left
```

Useful knobs:

- `--roboflow-model-id` (defaults to `crowd-counting-dataset-w3o7w/2`)
- `--people-confidence-threshold` (defaults to `0.35`)
- `--roboflow-api-key` (optional override for `ROBOFLOW_API_KEY`)

The OpenCV overlay will display both person boxes and a `People: N` counter.

---

## Training a dedicated cardboard-box CV model

A non-destructive training workflow (Roboflow + YOLOv8) is included here:

- `docs/CARDBOARD_MODEL_TRAINING.md`
- `training/cardboard_box_training.py`
- `requirements-training.txt`

Follow the guide for copy/paste installs, dataset download options, and training commands.

---

## How to extend this modular setup

Add new processors under `cv_modular/processors/` that implement:
- `process(frame) -> ProcessorResult`
- `close()`

Then register the new processor in `run_cv.py`.

That is all you need to keep scaling from finger counting to more CV capabilities (gestures, object detection, tracking, measurements, etc.).
