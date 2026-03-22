from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .interfaces import FrameProcessor, ProcessorResult


@dataclass
class PipelineOutput:
    frame: np.ndarray
    results: list[ProcessorResult]


class CVPipeline:
    """Runs processors in sequence on each frame."""

    def __init__(self, processors: list[FrameProcessor]) -> None:
        self.processors = processors

    def step(self, frame: np.ndarray) -> PipelineOutput:
        results: list[ProcessorResult] = []
        for processor in self.processors:
            result = processor.process(frame)
            results.append(result)
        return PipelineOutput(frame=frame, results=results)

    def close(self) -> None:
        for processor in self.processors:
            processor.close()


def run_webcam_loop(
    pipeline: CVPipeline,
    camera_index: int = 0,
    fallback_camera_indexes: list[int] | None = None,
    window_name: str = "Modular CV",
    on_output: Callable[[PipelineOutput], None] | None = None,
) -> None:
    candidate_indexes: list[int] = [camera_index]
    if fallback_camera_indexes:
        for index in fallback_camera_indexes:
            if index not in candidate_indexes:
                candidate_indexes.append(index)

    cap = None
    selected_index = None
    for index in candidate_indexes:
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            cap = candidate
            selected_index = index
            if index != camera_index:
                print(f"Primary webcam index={camera_index} unavailable. Using fallback index={index}.")
            break
        candidate.release()

    if cap is None:
        attempted = ", ".join(str(index) for index in candidate_indexes)
        raise RuntimeError(f"Unable to open webcam. Tried indexes: {attempted}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if selected_index is not None:
                    print(f"Webcam read failed for index={selected_index}.")
                break

            output = pipeline.step(frame)
            if on_output is not None:
                on_output(output)
            cv2.imshow(window_name, output.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()
