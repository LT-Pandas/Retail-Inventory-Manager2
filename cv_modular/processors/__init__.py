from .box_detector import BoxDetectorConfig, BoxDetectorProcessor
from .finger_counter import FingerCounterConfig, FingerCounterProcessor
from .hand_box_detector import HandBoxDetectorConfig, HandBoxDetectorProcessor

__all__ = [
    "BoxDetectorConfig",
    "BoxDetectorProcessor",
    "FingerCounterConfig",
    "FingerCounterProcessor",
    "HandBoxDetectorConfig",
    "HandBoxDetectorProcessor",
]

try:
    from .object_detector import ObjectDetectorConfig, ObjectDetectorProcessor
except Exception:  # pragma: no cover - optional dependency (MediaPipe)
    ObjectDetectorConfig = None
    ObjectDetectorProcessor = None
else:
    __all__.extend([
        "ObjectDetectorConfig",
        "ObjectDetectorProcessor",
    ])
