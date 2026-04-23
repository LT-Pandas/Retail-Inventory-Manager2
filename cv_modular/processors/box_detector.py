from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..interfaces import ProcessorResult


@dataclass
class BoxDetectorConfig:
    """Configuration for contour-based objectness detection."""

    min_area: int = 2500
    epsilon_ratio: float = 0.04
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_fill_ratio: float = 0.08
    draw_color: tuple[int, int, int] = (0, 255, 0)


class BoxDetectorProcessor:
    """Detect unknown objects by finding strong edge-bounded contours.

    This detector intentionally does not classify object type. It only identifies
    likely object regions and reports them as generic objects.
    """

    name = "box_detector"

    def __init__(self, config: BoxDetectorConfig | None = None) -> None:
        self.config = config or BoxDetectorConfig()

    def process(self, frame: np.ndarray) -> ProcessorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[dict[str, float | int | str]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approximation = cv2.approxPolyDP(contour, self.config.epsilon_ratio * perimeter, True)

            x, y, width, height = cv2.boundingRect(contour)
            if height == 0:
                continue

            aspect_ratio = width / float(height)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue

            fill_ratio = area / float(max(1, width * height))
            if fill_ratio < self.config.min_fill_ratio:
                continue

            cv2.drawContours(frame, [approximation], -1, self.config.draw_color, 2)
            label = f"Object {len(boxes) + 1}"
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
                    "label": "object",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": float(area),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "fill_ratio": round(fill_ratio, 3),
                }
            )

        cv2.putText(
            frame,
            f"Objects: {len(boxes)}",
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.config.draw_color,
            2,
        )

        return ProcessorResult(name=self.name, data={"boxes": boxes, "count": len(boxes)})

    def close(self) -> None:
        return None
