from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    processing_scale: float = 0.5
    shadow_min_saturation: int = 28
    shadow_min_value: int = 55
    draw_color: tuple[int, int, int] = (0, 255, 0)


class BoxDetectorProcessor:
    """Detect unknown objects by finding strong edge-bounded contours.

    This detector intentionally does not classify object type. It only identifies
    likely object regions and reports them as generic objects.
    """

    name = "box_detector"

    def __init__(self, config: BoxDetectorConfig | None = None) -> None:
        self.config = config or BoxDetectorConfig()
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=max(16, int(self.config.num_disparities / 16) * 16),
            blockSize=max(3, self.config.block_size | 1),
            P1=8 * 3 * (self.config.block_size**2),
            P2=32 * 3 * (self.config.block_size**2),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.left_map_x: np.ndarray | None = None
        self.left_map_y: np.ndarray | None = None
        self.right_map_x: np.ndarray | None = None
        self.right_map_y: np.ndarray | None = None
        self.calibration_loaded = False
        self._try_load_calibration()

    def _try_load_calibration(self) -> None:
        if not self.config.calibration_file:
            return
        calibration_path = Path(self.config.calibration_file).expanduser()
        if not calibration_path.exists():
            print(
                f"[box_detector] Stereo calibration file not found: {calibration_path}. "
                "Running without rectification."
            )
            return
        try:
            data = np.load(calibration_path, allow_pickle=True)
            key_aliases = {
                "left_map_x": ("left_map_x", "map1x", "leftMapX"),
                "left_map_y": ("left_map_y", "map1y", "leftMapY"),
                "right_map_x": ("right_map_x", "map2x", "rightMapX"),
                "right_map_y": ("right_map_y", "map2y", "rightMapY"),
            }
            extracted: dict[str, np.ndarray] = {}
            for canonical, aliases in key_aliases.items():
                for key in aliases:
                    if key in data:
                        extracted[canonical] = data[key]
                        break
            if len(extracted) == 4:
                self.left_map_x = extracted["left_map_x"]
                self.left_map_y = extracted["left_map_y"]
                self.right_map_x = extracted["right_map_x"]
                self.right_map_y = extracted["right_map_y"]
                self.calibration_loaded = True
                print(f"[box_detector] Loaded stereo rectification maps from {calibration_path}.")
                return
        except Exception as exc:
            print(f"[box_detector] Failed to load calibration file ({calibration_path}): {exc}")
            return

        print(
            f"[box_detector] Calibration file {calibration_path} does not include rectification maps. "
            "Running without rectification."
        )

    def _split_stereo_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = frame.shape[1]
        split = width // 2
        left = frame[:, :split].copy()
        right = frame[:, split : split + split].copy()
        return left, right

    def _rectify_if_available(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.calibration_loaded:
            return left, right
        rect_left = cv2.remap(left, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def process(self, frame: np.ndarray) -> ProcessorResult:
        scale = float(np.clip(self.config.processing_scale, 0.2, 1.0))
        if scale < 0.999:
            working = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            working = frame

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 180)

        saturation_mask = cv2.inRange(
            hsv,
            (0, self.config.shadow_min_saturation, self.config.shadow_min_value),
            (180, 255, 255),
        )
        bright_mask = cv2.inRange(hsv[:, :, 2], self.config.shadow_min_value + 20, 255)
        non_shadow_mask = cv2.bitwise_or(saturation_mask, bright_mask)
        edges = cv2.bitwise_and(edges, non_shadow_mask)

        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        candidates: list[dict[str, float | int | np.ndarray]] = []

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
            fill_ratio = area / float(max(1, width * height))
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            if fill_ratio < self.config.min_fill_ratio:
                continue

            fill_ratio = area / float(max(1, width * height))
            if fill_ratio < self.config.min_fill_ratio:
                continue

            if scale < 0.999:
                approximation = (approximation / scale).astype(np.int32)
                x = int(x / scale)
                y = int(y / scale)
                width = int(width / scale)
                height = int(height / scale)

            cv2.drawContours(frame, [approximation], -1, self.config.draw_color, 2)
            label = f"Object {len(boxes) + 1}"
            cv2.putText(
                overlay,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.draw_color,
                2,
            )
            mean_disparity = float(np.median(disparity[merged_mask > 0]))
            boxes.append(
                {
                    "label": "object",
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": float(area / (scale * scale)),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "fill_ratio": round(fill_ratio, 3),
                }
            )
            mask_list.append(merged_mask)

        split = frame.shape[1] // 2
        frame[:, :split] = overlay
        if not self.calibration_loaded:
            cv2.putText(
                frame,
                "Stereo not rectified (no calibration maps)",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
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

        return ProcessorResult(name=self.name, data={"boxes": boxes, "count": len(boxes), "masks": mask_list})

    def close(self) -> None:
        return None
