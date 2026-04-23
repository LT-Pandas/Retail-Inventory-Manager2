from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from picamera2 import Picamera2

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
    camera_width: int = 1920,
    camera_height: int = 1080,
    camera_fps: int = 30,
    camera_brightness: float = 0.08,
    camera_contrast: float = 1.2,
    camera_saturation: float = 1.15,
    camera_sharpness: float = 1.0,
    camera_autofocus: bool = True,
    camera_lens_position: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_key: Callable[[int], bool] | None = None,
    show_window: bool = True,
) -> None:
    candidate_indexes: list[int] = [camera_index]
    if fallback_camera_indexes:
        for index in fallback_camera_indexes:
            if index not in candidate_indexes:
                candidate_indexes.append(index)

    picam2 = None
    selected_index = None
    for index in candidate_indexes:
        candidate = None
        try:
            candidate = Picamera2(camera_num=index)
            config = candidate.create_preview_configuration(
                main={"format": "RGB888", "size": (camera_width, camera_height)}
            )
            candidate.configure(config)
            frame_duration_us = int(1_000_000 / max(1, camera_fps))
            candidate.set_controls(
                {
                    "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                    "AeEnable": True,
                    "AwbEnable": True,
                }
            )
            optional_controls = {
                "Brightness": camera_brightness,
                "Contrast": camera_contrast,
                "Saturation": camera_saturation,
                "Sharpness": camera_sharpness,
                "AfMode": 2 if camera_autofocus else 0,  # Continuous autofocus or manual focus mode.
            }
            if camera_lens_position is not None and not camera_autofocus:
                optional_controls["LensPosition"] = camera_lens_position

            for control_name, control_value in optional_controls.items():
                try:
                    candidate.set_controls({control_name: control_value})
                except Exception:
                    pass
            candidate.start()
            candidate.capture_array()
            picam2 = candidate
            selected_index = index
            if index != camera_index:
                print(f"Primary webcam index={camera_index} unavailable. Using fallback index={index}.")
            break
        except Exception:
            if candidate is not None:
                try:
                    candidate.stop()
                except Exception:
                    pass

    if picam2 is None:
        attempted = ", ".join(str(index) for index in candidate_indexes)
        raise RuntimeError(f"Unable to open webcam. Tried indexes: {attempted}")

    try:
        while True:
            if should_stop is not None and should_stop():
                break
            try:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                if selected_index is not None:
                    print(f"Webcam read failed for index={selected_index}.")
                break

            output = pipeline.step(frame)
            if on_output is not None:
                on_output(output)

            if show_window:
                cv2.imshow(window_name, output.frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if on_key is not None and key != 255:
                    if on_key(key):
                        break
    finally:
        picam2.stop()
        pipeline.close()
        if show_window:
            cv2.destroyAllWindows()
