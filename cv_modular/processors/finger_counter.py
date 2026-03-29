from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    # Allow direct execution (e.g. `python cv_modular/processors/finger_counter.py`).
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from cv_modular.interfaces import ProcessorResult
else:
    from ..interfaces import ProcessorResult


@dataclass
class FingerCounterConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6  # Kept for CLI compatibility.
    min_tracking_confidence: float = 0.5  # Kept for CLI compatibility.
    min_presence_confidence: float = 0.5  # Kept for CLI compatibility.
    assume_selfie_view: bool = True
    model_path: str | None = None  # Kept for backward compatibility; unused.
    min_hand_area: int = 3000


class FingerCounterProcessor:
    """OpenCV-based hand tracking + finger counting.

    This implementation avoids MediaPipe so it can run on Raspberry Pi Python 3.13
    environments where MediaPipe wheels may be unavailable.
    """

    name = "finger_counter"

    def __init__(self, config: FingerCounterConfig | None = None) -> None:
        self.config = config or FingerCounterConfig()

    def process(self, frame: np.ndarray) -> ProcessorResult:
        mask = self._build_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.putText(frame, "Fingers: 0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 0), 2)
            return ProcessorResult(name=self.name, data={"per_hand": [], "total": 0})

        hand_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(hand_contour)

        if contour_area < self.config.min_hand_area:
            cv2.putText(frame, "Fingers: 0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 0), 2)
            return ProcessorResult(name=self.name, data={"per_hand": [], "total": 0})

        finger_count = self._count_fingers_from_contour(hand_contour)
        x, y, w, h = cv2.boundingRect(hand_contour)

        cv2.drawContours(frame, [hand_contour], -1, (0, 200, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 200, 255), 2)
        cv2.putText(
            frame,
            f"Hand: {finger_count}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(frame, f"Fingers: {finger_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 0), 2)
        return ProcessorResult(name=self.name, data={"per_hand": [finger_count], "total": finger_count})

    def _build_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Broad skin-color range in YCrCb that works reasonably for varied lighting.
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def _count_fingers_from_contour(self, contour: np.ndarray) -> int:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0

        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return 1

        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i, 0]
            start = contour[s][0]
            end = contour[e][0]
            far = contour[f][0]

            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)

            if b == 0 or c == 0:
                continue

            angle = np.degrees(np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)))
            if angle < 90 and depth > 6000:
                finger_gaps += 1

        # Heuristic: number of extended fingers is gaps + 1, clipped to plausible range.
        return max(0, min(5, finger_gaps + 1))

    def close(self) -> None:
        return
