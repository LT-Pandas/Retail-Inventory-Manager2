# Retail Inventory Manager (Raspberry Pi + OpenCV)

This project runs real-time hand/finger counting and box detection on a Raspberry Pi camera feed, with optional outputs to:
- an SPI OLED display,
- an Arduino LED indicator over serial,
- BLE (nRF Connect).

---

## 1) Main code structure (used runtime components)

```text
.
├── run_cv.py                               # Main entrypoint + CLI options
├── cv_modular/
│   ├── pipeline.py                         # Picamera2 capture loop + processor pipeline
│   ├── finger_serial.py                    # Sends FINGERS:<n> over serial (Arduino)
│   ├── oled_display.py                     # SPI OLED output (luma.oled)
│   ├── ble_uint8.py                        # BLE uint8 GATT server (optional)
│   └── processors/
│       ├── hand_box_detector.py            # Combined hand + box processor
│       ├── finger_counter.py               # Hand tracking + finger counting
│       ├── box_detector.py                 # Contour-based rectangular box detection
│       └── object_detector.py              # Optional MediaPipe object detector
└── ArduinoFiveFingerAI.ino/
    └── ArduinoFiveFingerAI.ino.ino         # Arduino sketch for 5-LED count display
```

Notes:
- `run_cv.py` uses `HandBoxDetectorProcessor` by default.
- `ObjectDetectorProcessor` is optional and only used when `--detect-objects` (or `--object-model`) is passed.

---

## 2) Dependencies

### Required (from `requirements.txt`)

```txt
opencv-python>=4.9.0.80
numpy>=1.24.0
pyserial>=3.5
luma.oled>=3.13.0
luma.core>=2.4.2
```

### Installed separately only if needed
- `picamera2` (required on Raspberry Pi for camera capture).
- `gpiozero` (required only for `--button-controlled`).
- `bluezero` (required only for `--ble-enabled`).
- `mediapipe` (required only for `--detect-objects` / `--object-model`).

---

## 3) Hardware/components used

### Raspberry Pi side
- Raspberry Pi (with 40-pin header)
- Pi camera (used by `picamera2`)
- SPI OLED module (controller supported: `ssd1309`, `sh1107`, `ssd1327`)
- Momentary push button (optional start trigger)
- USB cable to Arduino (for serial LED output), or direct serial adapter if preferred

### Arduino side (optional LED bar)
- Arduino board
- 5x LEDs
- 5x current-limiting resistors (typical 220Ω to 330Ω)
- Jumper wires + breadboard

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

## C) Pi -> Arduino serial (optional)
This project typically uses USB serial:
- Connect Arduino to Pi via USB.
- Run with `--serial-port /dev/ttyACM0` (or your actual port).
- Message format sent by Pi: `FINGERS:<count>\n`.

## D) Arduino simple LED setup
Arduino sketch LED pins:
- D2, D3, D4, D5, D6 (5 LEDs total)

Wire each LED channel as:
- Arduino digital pin -> resistor (220Ω–330Ω) -> LED anode (+)
- LED cathode (-) -> GND

Behavior:
- If count is 3, LEDs on D2-D4 turn on; D5-D6 remain off.
- Count is clamped to range 0-5.

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
pip install gpiozero bluezero mediapipe
```

(Install only the ones you plan to use.)

## B) Standard run (camera + hand/box detection + OLED enabled by default)

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

Send count to Arduino serial:

```bash
python run_cv.py --serial-port /dev/ttyACM0 --serial-baud 115200
```

Disable OLED output:

```bash
python run_cv.py --no-oled-enabled
```

Enable BLE uint8 server:

```bash
python run_cv.py --ble-enabled --ble-adapter-address B8:27:EB:00:00:01
```

Enable optional MediaPipe object detection:

```bash
python run_cv.py --detect-objects
```

---

## 6) Runtime outputs (what updates where)
- OpenCV window overlay:
  - Finger count (`Fingers: N`)
  - Box count (`Boxes: N`)
- OLED: shows the latest finger count as a single large number.
- Serial: sends `FINGERS:<n>` only when value changes (rate-limited).
- BLE: exposes current count as uint8 characteristic; notify using key `o`.

---

## 7) Quick troubleshooting
- Camera fails to open: try `--camera-index 0 --fallback-camera-indexes 1`.
- OLED init fails: try explicit `--oled-driver sh1107` (or `ssd1309` / `ssd1327`).
- Button not responding: verify BCM numbering and physical wiring (GPIO17 is physical pin 11).
- Serial not updating LEDs: confirm baud is `115200` and incoming text is `FINGERS:<n>`.
