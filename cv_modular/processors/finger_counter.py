from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

if __package__ in (None, ""):
    # Allow direct execution (e.g. `python cv_modular/processors/finger_counter.py`).
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from cv_modular.interfaces import ProcessorResult
else:
    from ..interfaces import ProcessorResult


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class FingerCounterConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    assume_selfie_view: bool = True
    model_path: str | None = None


def _resolve_hand_landmarker_model(config: FingerCounterConfig) -> str:
    if config.model_path:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {model_path}")
        return str(model_path)

    cache_dir = Path.home() / ".cache" / "retail-inventory-manager"
    cache_dir.mkdir(parents=True, exist_ok=True)
    default_model = cache_dir / "hand_landmarker.task"

    if not default_model.exists():
        urlretrieve(MODEL_URL, default_model)

    return str(default_model)


class FingerCounterProcessor:
    name = "finger_counter"

    def __init__(self, config: FingerCounterConfig | None = None) -> None:
        self.config = config or FingerCounterConfig()
        model_path = _resolve_hand_landmarker_model(self.config)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> ProcessorResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        counts: list[int] = []

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[idx][0].category_name if idx < len(result.handedness) else "Unknown"
            finger_count = self._count_extended_fingers(hand_landmarks, handedness)
            counts.append(finger_count)

            self._draw_hand_landmarks(frame, hand_landmarks)
            wrist = hand_landmarks[0]
            h, w = frame.shape[:2]
            origin = (int(wrist.x * w), int(wrist.y * h) - 20)
            cv2.putText(
                frame,
                f"{handedness}: {finger_count}",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        total = max(counts) if counts else 0
        cv2.putText(frame, f"Fingers: {total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 0), 2)
        return ProcessorResult(name=self.name, data={"per_hand": counts, "total": total})

    def _draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks) -> None:
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start_idx, end_idx in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 200, 255), 2, cv2.LINE_AA)

        for x, y in points:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _count_extended_fingers(self, hand_landmarks, handedness_label: str) -> int:
        finger_tip_ids = [8, 12, 16, 20]
        finger_pip_ids = [6, 10, 14, 18]

        count = 0
        for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
            if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y:
                count += 1

        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        is_right = handedness_label.lower() == "right"

        if self.config.assume_selfie_view:
            thumb_extended = thumb_tip.x < thumb_ip.x if is_right else thumb_tip.x > thumb_ip.x
        else:
            thumb_extended = thumb_tip.x > thumb_ip.x if is_right else thumb_tip.x < thumb_ip.x

        if thumb_extended:
            count += 1

        return count

    def close(self) -> None:
        self.landmarker.close()
