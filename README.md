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

### Improve camera image quality (resolution + tuning)

The default camera profile is tuned for a clearer feed (`1280x720 @ 30 FPS`) with mild brightness/contrast/saturation/sharpness boosts.

```bash
python run_cv.py \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --camera-brightness 0.08 \
  --camera-contrast 1.2 \
  --camera-saturation 1.15 \
  --camera-sharpness 1.35
```

If the image looks oversharpened or noisy in your lighting, reduce `--camera-sharpness` first.

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




## OLED output on Raspberry Pi (SPI)

You can mirror the live finger count to an SPI OLED (for example SH1107 / SSD1309 / SSD1327 via `luma.oled`).

Install dependencies:

```bash
pip install -r requirements.txt
```

OLED output is enabled by default now, so pressing the play/run button on `run_cv.py` will automatically initialize the display (auto driver fallback: `ssd1309` -> `sh1107` -> `ssd1327`):

```bash
python run_cv.py
```

If you ever want to run without OLED output:

```bash
python run_cv.py --no-oled-enabled
```

On startup, a test message is now shown for 2 seconds by default (`RPI5 OLED OK`) so you can quickly confirm the display wiring before finger counts begin.

Choose a specific driver:

```bash
python run_cv.py --oled-enabled --oled-driver sh1107
```

If your display only works with another controller:

```bash
python run_cv.py --oled-enabled --oled-driver ssd1309
# or
python run_cv.py --oled-enabled --oled-driver ssd1327
```

Customize or disable the startup test message:

```bash
python run_cv.py --oled-enabled --oled-test-message "OLED TEST OK" --oled-test-seconds 3
# disable startup message
python run_cv.py --oled-enabled --oled-test-message ""
```

SPI/GPIO pins are configurable if your wiring differs:

```bash
python run_cv.py --oled-enabled --oled-spi-port 0 --oled-spi-device 0 --oled-gpio-dc 25 --oled-gpio-rst 24
```

The OLED output is a single number that updates only when the count changes.

---

## Start/stop with a physical button (GPIO 17 + GND pin 14)

You can run the app in button-controlled mode so it only starts after a button press and exits on the next press.

Wiring:
- One button leg -> **BCM GPIO 17** (physical pin 11)
- Other button leg -> **GND physical pin 14**

Run:

```bash
python run_cv.py --button-controlled --button-gpio-pin 17
```

Behavior:
- **First press**: starts the CV loop.
- **Second press**: cleanly stops the CV loop and exits the program.

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
