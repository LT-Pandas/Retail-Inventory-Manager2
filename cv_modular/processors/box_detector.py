from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..interfaces import ProcessorResult


@dataclass
class BoxDetectorConfig:
    """Configuration for geometric box detection using contours."""

    min_area: int = 2500
    epsilon_ratio: float = 0.04
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.2
    draw_color: tuple[int, int, int] = (0, 255, 0)


class BoxDetectorProcessor:
    """Detect cardboard-like rectangular boxes using contour approximation.

    This model-free detector is useful when you only need to identify box shapes
    and do not have a trained object-detection model available.
    """

    name = "box_detector"

    def __init__(self, config: BoxDetectorConfig | None = None) -> None:
        self.config = config or BoxDetectorConfig()

    def process(self, frame: np.ndarray) -> ProcessorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[dict[str, float | int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, self.config.epsilon_ratio * perimeter, True)
            if len(approximation) != 4:
                continue

            x, y, width, height = cv2.boundingRect(approximation)
            if height == 0:
                continue

            aspect_ratio = width / float(height)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue

            cv2.drawContours(frame, [approximation], -1, self.config.draw_color, 2)
            label = f"Box {len(boxes) + 1}"
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.draw_color,
                2,
            )

            boxes.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": float(area),
                    "aspect_ratio": round(aspect_ratio, 3),
                }
            )

        cv2.putText(
            frame,
            f"Boxes: {len(boxes)}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.draw_color,
            2,
        )

        return ProcessorResult(name=self.name, data={"boxes": boxes, "count": len(boxes)})

    def close(self) -> None:
        return None
