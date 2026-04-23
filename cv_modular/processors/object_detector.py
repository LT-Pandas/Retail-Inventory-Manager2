from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from ..interfaces import ProcessorResult


MODEL_URLS: dict[str, str] = {
    "efficientdet_lite0": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
    ),
    "efficientdet_lite2": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite2/int8/1/efficientdet_lite2.tflite"
    ),
}


@dataclass
class ObjectDetectorConfig:
    model_path: str | None = None
    model_variant: str = "efficientdet_lite0"
    max_results: int = 5
    score_threshold: float = 0.25
    label_filter: tuple[str, ...] | None = None
    draw_count: bool = True


def _resolve_object_detector_model(config: ObjectDetectorConfig) -> str:
    if config.model_path:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Object detector model not found: {model_path}")
        return str(model_path)

    if config.model_variant not in MODEL_URLS:
        variants = ", ".join(sorted(MODEL_URLS))
        raise ValueError(
            f"Unsupported model variant {config.model_variant!r}. "
            f"Supported variants: {variants}."
        )

    cache_dir = Path.home() / ".cache" / "retail-inventory-manager"
    cache_dir.mkdir(parents=True, exist_ok=True)
    default_model = cache_dir / f"{config.model_variant}.tflite"

    if not default_model.exists():
        urlretrieve(MODEL_URLS[config.model_variant], default_model)

    return str(default_model)


class ObjectDetectorProcessor:
    """Optional object detector plugin using MediaPipe Tasks.

    Uses a provided .tflite model path, or auto-downloads a default model.
    """

    name = "object_detector"

    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config
        self.model_path = _resolve_object_detector_model(self.config)
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            max_results=self.config.max_results,
            score_threshold=self.config.score_threshold,
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

    def process(self, frame: np.ndarray) -> ProcessorResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self.detector.detect(mp_image)

        labels: list[str] = []
        for detection in detection_result.detections:
            category = detection.categories[0]
            category_name = category.category_name.lower()
            if self.config.label_filter and not any(
                label in category_name for label in self.config.label_filter
            ):
                continue

            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 255), 2)

            label = f"{category.category_name}:{category.score:.2f}"
            labels.append(label)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 2)

        count = len(labels)
        if self.config.draw_count:
            cv2.putText(
                frame,
                f"Objects: {count}",
                (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 200, 255),
                2,
            )

        return ProcessorResult(name=self.name, data={"detections": labels, "count": count})

    def close(self) -> None:
        self.detector.close()
