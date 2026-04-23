# Retail Inventory Manager (Raspberry Pi + OpenCV)

This project runs real-time object detection on a Raspberry Pi camera feed, with optional outputs to:
- an SPI OLED display,
- BLE (nRF Connect).

---

## 1) Main code structure (used runtime components)

```text
.
├── run_cv.py                               # Main entrypoint + CLI options
├── cv_modular/
│   ├── pipeline.py                         # Picamera2 capture loop + processor pipeline
│   ├── oled_display.py                     # SPI OLED output (luma.oled)
│   ├── ble_uint8.py                        # BLE uint8 GATT server (optional)
│   └── processors/
│       └── box_detector.py                 # OpenCV contour-based object detector
└── docs/
    └── CARDBOARD_MODEL_TRAINING.md         # Optional model-training workflow docs
```

Notes:
- `run_cv.py` uses OpenCV and now supports stereo depth-aware mask generation from a dual-camera feed.

---

## 2) Dependencies

### Required (from `requirements.txt`)

```txt
opencv-python>=4.9.0.80
numpy>=1.24.0
luma.oled>=3.13.0
luma.core>=2.4.2
```

### Installed separately only if needed
- `picamera2` (required on Raspberry Pi for camera capture).
- `gpiozero` (required only for `--button-controlled`).
- `bluezero` (required only for `--ble-enabled`).

## D) Arduino simple LED setup
Arduino sketch LED pins:
- D2, D3, D4, D5, D6 (5 LEDs total)

## 3) Hardware/components used

- Raspberry Pi (with 40-pin header)
- Pi camera (used by `picamera2`)
- SPI OLED module (controller supported: `ssd1309`, `sh1107`, `ssd1327`)
- Momentary push button (optional start trigger)

---

## 4) Pin mapping and wiring

## A) Raspberry Pi -> OLED (SPI)
Default runtime pins in this project:
- `--oled-spi-port 0 --oled-spi-device 0`
- `--oled-gpio-dc 25`
- `--oled-gpio-rst 24`

Recommended wiring (Pi 40-pin header):

| OLED signal | Pi BCM | Pi physical pin |
|---|---:|---:|
| VCC | 3.3V | 1 (or 17) |
| GND | GND | 6 (or any GND) |
| SCLK / CLK | GPIO11 (SPI0 SCLK) | 23 |
| MOSI / DIN | GPIO10 (SPI0 MOSI) | 19 |
| CS | GPIO8 (SPI0 CE0) | 24 |
| DC | GPIO25 | 22 |
| RST / RES | GPIO24 | 18 |

Driver chip note:
- Your OLED module’s onboard controller/driver chip should be one of: `ssd1309`, `sh1107`, or `ssd1327`.
- Default auto-init tries: `ssd1309` -> `sh1107` -> `ssd1327`.

## B) Raspberry Pi button input (optional)
Project default:
- `--button-gpio-pin 17` (BCM numbering)

Wiring:
- Button leg 1 -> **BCM 17** (physical pin **11**)
- Button leg 2 -> **GND** (for example physical pin **14**)

Important:
- Do **not** use physical pin 17 for this signal (that pin is 3.3V power).

---

## 5) How to run

## A) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install optional packages only when needed:

```bash
pip install gpiozero bluezero
```

(Install only the ones you plan to use.)

## B) Standard run (camera + object detection + OLED enabled by default)

```bash
python run_cv.py
```

Controls:
- Press `q` to quit.
- Press `o` to send current count as BLE notification (when BLE is enabled).

## C) Common run modes

Button-controlled startup:

```bash
python run_cv.py --button-controlled --button-gpio-pin 17
```

Disable OLED output:

```bash
python run_cv.py --no-oled-enabled
```

Enable BLE uint8 server:

```bash
python run_cv.py --ble-enabled --ble-adapter-address B8:27:EB:00:00:01
```

Object detection is enabled by default:

```bash
python run_cv.py
```

Tune contour sensitivity for your setup:

```bash
python run_cv.py --min-object-area 1800 --object-epsilon-ratio 0.03
```

Enable depth-informed merging of nearby color blobs (single physical object with multiple colors):

```bash
python run_cv.py \
  --stereo-calibration-file stereo_calibration.npz \
  --stereo-disparity-threshold 1.2 \
  --depth-merge-threshold 4.0 \
  --stereo-bbox-merge-gap 42
```

Constrain object shape if needed:

```bash
python run_cv.py --object-min-aspect-ratio 0.6 --object-max-aspect-ratio 1.8
```

Reduce shadow false positives and improve FPS:

```bash
python run_cv.py --object-processing-scale 0.45 --shadow-min-saturation 35 --shadow-min-value 65
```

---

## 6) Runtime outputs (what updates where)
- OpenCV window overlay:
  - Object count (`Objects: N`)
  - A closed bounding frame for each detected object.
- OLED: shows the latest object count as a single large number.
- BLE: exposes current object count as uint8 characteristic; notify using key `o`.

---

## 7) Quick troubleshooting
- Dual-camera stream requires two working indexes, e.g. `--camera-index 0 --fallback-camera-indexes 1`.
- If the window shows `Stereo not rectified (no calibration maps)`, provide `--stereo-calibration-file <file>.npz` containing left/right rectification maps.
- OLED init fails: try explicit `--oled-driver sh1107` (or `ssd1309` / `ssd1327`).
- Button not responding: verify BCM numbering and physical wiring (GPIO17 is physical pin 11).
- Object detection poor quality:
  - Increase `--min-object-area` if noise/small contours are getting counted.
  - Lower `--min-object-area` if valid objects are being missed.
  - Tighten `--object-min-aspect-ratio` / `--object-max-aspect-ratio` to match expected object shape.
