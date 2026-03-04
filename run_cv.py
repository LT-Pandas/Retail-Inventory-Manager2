from __future__ import annotations

import argparse

from cv_modular import CVPipeline, run_webcam_loop
from cv_modular.finger_serial import FingerSerialSender, FingerSerialSenderConfig
from cv_modular.processors import (
    BoxDetectorConfig,
    FingerCounterConfig,
    HandBoxDetectorConfig,
    HandBoxDetectorProcessor,
    ObjectDetectorConfig,
    ObjectDetectorProcessor
)


def _extract_finger_total(results) -> int | None:
    for result in results:
        if result.name == "hand_box_detector":
            return result.data.get("hands", {}).get("total")
        if result.name == "finger_counter":
            return result.data.get("total")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular MediaPipe CV demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.6)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-presence-confidence", type=float, default=0.5)
    parser.add_argument(
        "--hand-model",
        type=str,
        default=None,
        help="Optional path to a MediaPipe hand_landmarker.task model.",
    )
    parser.add_argument(
        "--no-assume-selfie-view",
        action="store_true",
        help="Use this when the incoming image is not mirrored.",
    )
    parser.add_argument(
        "--detect-objects",
        action="store_true",
        help="Enable MediaPipe object detection (auto-downloads default model).",
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

    if args.detect_objects or args.object_model:
        processors.append(ObjectDetectorProcessor(ObjectDetectorConfig(model_path=args.object_model)))
    pipeline = CVPipeline(processors)

    sender = None
    if args.serial_port:
        sender = FingerSerialSender(FingerSerialSenderConfig(port=args.serial_port, baud=args.serial_baud))

    def on_output(output) -> None:
        if sender is None:
            return
        total = _extract_finger_total(output.results)
        if total is None:
            return
        print("Sending finger total:", total)
        sender.send_finger_count(total)

    try:
        run_webcam_loop(pipeline, camera_index=args.camera_index, on_output=on_output)
    finally:
        if sender is not None:
            sender.close()


if __name__ == "__main__":
    main()
