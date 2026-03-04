from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import ProcessorResult
from .box_detector import BoxDetectorConfig, BoxDetectorProcessor
from .finger_counter import FingerCounterConfig, FingerCounterProcessor


@dataclass
class HandBoxDetectorConfig:
    """Single processor config for hand + box detection."""

    hand: FingerCounterConfig
    box: BoxDetectorConfig


class HandBoxDetectorProcessor:
    """Runs hand landmark detection and geometric box detection together."""

    name = "hand_box_detector"

    def __init__(self, config: HandBoxDetectorConfig) -> None:
        self.config = config
        self.hand_detector = FingerCounterProcessor(config.hand)
        self.box_detector = BoxDetectorProcessor(config.box)

    def process(self, frame: np.ndarray) -> ProcessorResult:
        hand_result = self.hand_detector.process(frame)
        box_result = self.box_detector.process(frame)

        return ProcessorResult(
            name=self.name,
            data={
                "hands": hand_result.data,
                "boxes": box_result.data,
            },
        )

    def close(self) -> None:
        self.hand_detector.close()
        self.box_detector.close()

