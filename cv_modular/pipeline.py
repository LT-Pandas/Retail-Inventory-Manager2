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
) -> None:
    candidate_indexes: list[int] = [camera_index]
    if fallback_camera_indexes:
        for index in fallback_camera_indexes:
            if index not in candidate_indexes:
                candidate_indexes.append(index)
    if len(candidate_indexes) < 2:
        raise RuntimeError(
            "Dual-camera mode requires two unique indexes. "
            "Provide --camera-index and at least one --fallback-camera-indexes value."
        )

    selected_indexes = candidate_indexes[:2]
    cameras: list[Picamera2] = []
    opened_indexes: list[int] = []
    for index in selected_indexes:
        camera = None
        try:
            camera = Picamera2(camera_num=index)
            config = camera.create_preview_configuration(
                main={"format": "RGB888", "size": (camera_width, camera_height)}
            )
            camera.configure(config)
            frame_duration_us = int(1_000_000 / max(1, camera_fps))
            camera.set_controls(
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
                    camera.set_controls({control_name: control_value})
                except Exception:
                    pass
            camera.start()
            camera.capture_array()
            cameras.append(camera)
            opened_indexes.append(index)
        except Exception:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

    if len(cameras) != 2:
        attempted = ", ".join(str(index) for index in selected_indexes)
        opened = ", ".join(str(index) for index in opened_indexes) or "none"
        for camera in cameras:
            try:
                camera.stop()
            except Exception:
                pass
        raise RuntimeError(
            f"Unable to open two webcams. Requested indexes: {attempted}; opened: {opened}."
        )
    print(f"Dual-camera stream active with indexes={opened_indexes[0]},{opened_indexes[1]}.")

    try:
        while True:
            if should_stop is not None and should_stop():
                break
            try:
                frames_bgr: list[np.ndarray] = []
                for camera in cameras:
                    frame = camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames_bgr.append(frame)
            except Exception:
                print("Webcam read failed for one of the active cameras.")
                break

            # Keep frames crisp by avoiding interpolation-based resizing.
            # If camera outputs differ slightly in size, center-crop to the common height.
            min_height = min(frame.shape[0] for frame in frames_bgr)
            normalized_frames = []
            for frame in frames_bgr:
                if frame.shape[0] != min_height:
                    delta = frame.shape[0] - min_height
                    top = max(0, delta // 2)
                    bottom = top + min_height
                    frame = frame[top:bottom, :]
                normalized_frames.append(frame)

            stitched_frame = cv2.hconcat(normalized_frames)
            output = pipeline.step(stitched_frame)
            if on_output is not None:
                on_output(output)
            cv2.imshow(window_name, output.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if on_key is not None and key != 255:
                if on_key(key):
                    break
    finally:
        for camera in cameras:
            try:
                camera.stop()
            except Exception:
                pass
        pipeline.close()
        cv2.destroyAllWindows()
