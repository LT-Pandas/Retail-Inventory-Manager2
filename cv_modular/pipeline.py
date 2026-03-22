from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
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


class _Picamera2Capture:
    """Small adapter that matches OpenCV VideoCapture's read/release API."""

    def __init__(self, camera_index: int) -> None:
        from picamera2 import Picamera2

        self.camera_index = camera_index
        self.picam = Picamera2(camera_num=camera_index)
        preview_config = self.picam.create_preview_configuration(main={"format": "RGB888"})
        self.picam.configure(preview_config)
        self.picam.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame_rgb = self.picam.capture_array()
        if frame_rgb is None:
            return False, None
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self) -> None:
        self.picam.stop()
        self.picam.close()


def _open_video_capture(index: int):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap, f"opencv:{index}"
    cap.release()
    return None, None


def _open_picamera_capture(index: int):
    if find_spec("picamera2") is None:
        return None, None

    try:
        cap = _Picamera2Capture(index)
        return cap, f"picamera2:{index}"
    except Exception as exc:
        print(f"Unable to initialize picamera2 index={index}: {exc}")
        return None, None


def run_webcam_loop(
    pipeline: CVPipeline,
    camera_index: int = 0,
    fallback_camera_indexes: list[int] | None = None,
    allow_picamera2_fallback: bool = True,
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
        candidate, backend = _open_video_capture(index)
        if candidate is None and allow_picamera2_fallback:
            candidate, backend = _open_picamera_capture(index)

        if candidate is not None:
            cap = candidate
            selected_index = index
            if index != camera_index:
                print(
                    f"Primary webcam index={camera_index} unavailable. "
                    f"Using fallback index={index} via {backend}."
                )
            elif backend is not None:
                print(f"Using camera index={index} via {backend}.")
            break

    if cap is None:
        attempted = ", ".join(str(index) for index in candidate_indexes)
        raise RuntimeError(
            "Unable to open webcam. Tried indexes: "
            f"{attempted}. If you are on Raspberry Pi, install/enable picamera2 (libcamera)."
        )

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
