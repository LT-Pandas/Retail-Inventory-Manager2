from __future__ import annotations

import argparse

from cv_modular import CVPipeline, run_webcam_loop
from cv_modular.finger_serial import FingerSerialSender, FingerSerialSenderConfig
from cv_modular.oled_display import OledCountDisplay, OledDisplayConfig
from cv_modular.processors import (
    BoxDetectorConfig,
    FingerCounterConfig,
    HandBoxDetectorConfig,
    HandBoxDetectorProcessor,
    ObjectDetectorConfig,
    ObjectDetectorProcessor,
)


def _extract_finger_total(results) -> int | None:
    for result in results:
        if result.name == "hand_box_detector":
            return result.data.get("hands", {}).get("total")
        if result.name == "finger_counter":
            return result.data.get("total")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular OpenCV CV demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--fallback-camera-indexes",
        type=int,
        nargs="*",
        default=[1],
        help="Fallback camera indexes to try if --camera-index fails (default: 1).",
    )
    parser.add_argument("--max-num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument(
        "--hand-model",
        type=str,
        default=None,
        help="Deprecated: kept for backward compatibility; OpenCV hand tracking ignores this.",
    )
    parser.add_argument(
        "--no-assume-selfie-view",
        action="store_true",
        help="Use this when the incoming image is not mirrored.",
    )
    parser.add_argument(
        "--detect-objects",
        action="store_true",
        help="Enable optional MediaPipe object detection (auto-downloads default model).",
    )
    parser.add_argument(
        "--object-model",
        type=str,
        default=None,
        help="Optional path to an object detection .tflite model. If omitted, a default model is downloaded.",
    )
    parser.add_argument(
        "--box-min-area",
        type=int,
        default=2500,
        help="Minimum contour area for a candidate box.",
    )
    parser.add_argument("--camera-width", type=int, default=1920, help="Camera capture width in pixels.")
    parser.add_argument("--camera-height", type=int, default=1080, help="Camera capture height in pixels.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Target camera FPS.")
    parser.add_argument(
        "--camera-brightness",
        type=float,
        default=0.08,
        help="Camera brightness control; small positive values can lift shadows.",
    )
    parser.add_argument("--camera-contrast", type=float, default=1.2, help="Camera contrast control.")
    parser.add_argument("--camera-saturation", type=float, default=1.15, help="Camera saturation control.")
    parser.add_argument("--camera-sharpness", type=float, default=1.0, help="Camera sharpness control.")
    parser.add_argument(
        "--no-camera-autofocus",
        action="store_true",
        help="Disable camera autofocus if your module supports focus controls.",
    )
    parser.add_argument(
        "--camera-lens-position",
        type=float,
        default=None,
        help="Optional manual lens position (requires --no-camera-autofocus and supported camera hardware).",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default=None,
        help="Optional serial port for Arduino output (e.g. COM5 or /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=115200,
        help="Baud rate for serial finger-count messages.",
    )
    parser.add_argument(
        "--oled-enabled",
        action="store_true",
        help="Enable OLED SPI output for finger count.",
    )
    parser.add_argument(
        "--oled-driver",
        choices=["auto", "sh1107", "ssd1309", "ssd1327"],
        default="auto",
        help="OLED driver to use. Default auto tries ssd1309, sh1107, then ssd1327.",
    )
    parser.add_argument("--oled-spi-port", type=int, default=0, help="OLED SPI port index.")
    parser.add_argument("--oled-spi-device", type=int, default=0, help="OLED SPI chip-select device index.")
    parser.add_argument("--oled-gpio-dc", type=int, default=25, help="OLED DC GPIO pin.")
    parser.add_argument("--oled-gpio-rst", type=int, default=24, help="OLED reset GPIO pin.")
    parser.add_argument(
        "--oled-test-message",
        type=str,
        default="RPI5 OLED OK",
        help="Startup test text shown once when OLED is enabled.",
    )
    parser.add_argument(
        "--oled-test-seconds",
        type=float,
        default=2.0,
        help="How long to display the startup OLED test text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    processors = [
        HandBoxDetectorProcessor(
            HandBoxDetectorConfig(
                hand=FingerCounterConfig(
                    max_num_hands=args.max_num_hands,
                    min_detection_confidence=args.min_detection_confidence,
                    min_tracking_confidence=args.min_tracking_confidence,
                    min_presence_confidence=args.min_presence_confidence,
                    assume_selfie_view=not args.no_assume_selfie_view,
                    model_path=args.hand_model,
                ),
                box=BoxDetectorConfig(min_area=args.box_min_area),
            )
        )
    ]

    if (args.detect_objects or args.object_model) and (ObjectDetectorProcessor is None or ObjectDetectorConfig is None):
        raise RuntimeError(
            "Object detection requires MediaPipe, which is not installed. "
            "Disable --detect-objects or install mediapipe."
        )

    if args.detect_objects or args.object_model:
        processors.append(ObjectDetectorProcessor(ObjectDetectorConfig(model_path=args.object_model)))
    pipeline = CVPipeline(processors)

    sender = None
    if args.serial_port:
        sender = FingerSerialSender(FingerSerialSenderConfig(port=args.serial_port, baud=args.serial_baud))

    oled_display = None
    if args.oled_enabled:
        oled_display = OledCountDisplay(
            OledDisplayConfig(
                spi_port=args.oled_spi_port,
                spi_device=args.oled_spi_device,
                gpio_dc=args.oled_gpio_dc,
                gpio_rst=args.oled_gpio_rst,
                driver=args.oled_driver,
            )
        )
        print(f"OLED display enabled using driver: {oled_display.active_driver}")
        if args.oled_test_message:
            oled_display.render_message(args.oled_test_message)
            print(f"OLED test message: {args.oled_test_message!r}")
            if args.oled_test_seconds > 0:
                time.sleep(args.oled_test_seconds)

    def on_output(output) -> None:
        total = _extract_finger_total(output.results)
        if total is None:
            return

        if sender is not None:
            print("Sending count:", total)
            sender.send_finger_count(total)

        if oled_display is not None:
            oled_display.render_count(total)

    try:
        run_webcam_loop(
            pipeline,
            camera_index=args.camera_index,
            fallback_camera_indexes=args.fallback_camera_indexes,
            on_output=on_output,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            camera_brightness=args.camera_brightness,
            camera_contrast=args.camera_contrast,
            camera_saturation=args.camera_saturation,
            camera_sharpness=args.camera_sharpness,
            camera_autofocus=not args.no_camera_autofocus,
            camera_lens_position=args.camera_lens_position,
        )
    finally:
        if sender is not None:
            sender.close()
        if oled_display is not None:
            oled_display.close()


if __name__ == "__main__":
    main()
