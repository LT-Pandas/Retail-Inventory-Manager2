from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

from ..interfaces import ProcessorResult


@dataclass
class PeopleCounterConfig:
    """Configuration for Roboflow-powered people counting."""

    model_id: str = "crowd-counting-dataset-w3o7w/2"
    api_url: str = "https://detect.roboflow.com"
    api_key: str | None = None
    confidence_threshold: float = 0.35
    stereo_layout: str = "none"  # one of: none, left-right, right-left
    stereo_eye: str = "left"  # one of: left, right


class PeopleCounterProcessor:
    """Detect and count people using a Roboflow hosted model.

    For stereo side-by-side streams, this processor can run inference on a single
    eye and project bounding boxes back onto the original frame.
    """

    name = "people_counter"

    def __init__(self, config: PeopleCounterConfig | None = None) -> None:
        self.config = config or PeopleCounterConfig()

        api_key = self.config.api_key or os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "Roboflow API key missing. Set ROBOFLOW_API_KEY or pass --roboflow-api-key."
            )

        self.client = InferenceHTTPClient(api_url=self.config.api_url, api_key=api_key)

    def process(self, frame: np.ndarray) -> ProcessorResult:
        inference_frame, offset_x = self._select_inference_frame(frame)

        result = self.client.infer(inference_frame, model_id=self.config.model_id)
        predictions = result.get("predictions", [])

        people: list[dict[str, float | int]] = []
        for pred in predictions:
            class_name = str(pred.get("class", "")).lower()
            confidence = float(pred.get("confidence", 0.0))
            if class_name != "person" or confidence < self.config.confidence_threshold:
                continue

            width = int(pred["width"])
            height = int(pred["height"])
            x_center = int(pred["x"])
            y_center = int(pred["y"])

            x1 = x_center - width // 2 + offset_x
            y1 = y_center - height // 2
            x2 = x_center + width // 2 + offset_x
            y2 = y_center + height // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 80, 255), 2)
            cv2.putText(
                frame,
                f"Person {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 80, 255),
                2,
            )

            people.append(
                {
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "confidence": round(confidence, 4),
                }
            )

        cv2.putText(
            frame,
            f"People: {len(people)}",
            (12, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 80, 255),
            2,
        )

        return ProcessorResult(name=self.name, data={"count": len(people), "people": people})

    def _select_inference_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        if self.config.stereo_layout == "none":
            return frame, 0

        height, width = frame.shape[:2]
        midpoint = width // 2

        if self.config.stereo_layout not in {"left-right", "right-left"}:
            raise ValueError(
                f"Unsupported stereo layout '{self.config.stereo_layout}'. "
                "Expected one of: none, left-right, right-left"
            )

        if self.config.stereo_layout == "left-right":
            left_eye = frame[:, :midpoint]
            right_eye = frame[:, midpoint:]
        else:  # right-left
            right_eye = frame[:, :midpoint]
            left_eye = frame[:, midpoint:]

        if self.config.stereo_eye == "left":
            return left_eye, 0 if self.config.stereo_layout == "left-right" else midpoint
        if self.config.stereo_eye == "right":
            return right_eye, midpoint if self.config.stereo_layout == "left-right" else 0

        raise ValueError(
            f"Unsupported stereo eye '{self.config.stereo_eye}'. Expected 'left' or 'right'."
        )

    def close(self) -> None:
        return
