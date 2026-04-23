from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..interfaces import ProcessorResult


@dataclass
class BoxDetectorConfig:
    """Configuration for stereo-aware objectness detection."""

    min_area: int = 2500
    epsilon_ratio: float = 0.04
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_fill_ratio: float = 0.08
    # Stereo/depth controls.
    min_disparity: int = 0
    num_disparities: int = 96
    block_size: int = 7
    disparity_foreground_threshold: float = 1.0
    min_depth_contrast: float = 0.75
    color_saturation_threshold: int = 35
    color_value_threshold: int = 55
    min_edge_ratio: float = 0.008
    depth_merge_threshold: float = 4.0
    bbox_gap_merge_px: int = 42
    mask_alpha: float = 0.28
    processing_scale: float = 0.6
    calibration_file: str | None = "stereo_calibration.npz"
    draw_color: tuple[int, int, int] = (0, 255, 0)


class BoxDetectorProcessor:
    """Detect unknown objects with stereo depth + color continuity.

    The processor expects a side-by-side frame (left | right) from two cameras.
    It creates a depth-informed foreground mask on the left image and merges
    nearby color blobs when their depth is continuous.
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
        left, right = self._split_stereo_frame(frame)
        left, right = self._rectify_if_available(left, right)
        scale = min(1.0, max(0.25, float(self.config.processing_scale)))
        if scale < 1.0:
            left_proc = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right_proc = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            left_proc = left
            right_proc = right

        gray_left = cv2.cvtColor(left_proc, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_proc, cv2.COLOR_BGR2GRAY)

        disparity_raw = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity = cv2.medianBlur(disparity_raw, 5)
        valid_disparity = disparity > self.config.disparity_foreground_threshold

        hsv_left = cv2.cvtColor(left_proc, cv2.COLOR_BGR2HSV)
        colorful = (
            (hsv_left[:, :, 1] >= self.config.color_saturation_threshold)
            & (hsv_left[:, :, 2] >= self.config.color_value_threshold)
        )
        edge_map = cv2.Canny(gray_left, 50, 150)

        foreground = np.zeros_like(gray_left, dtype=np.uint8)
        # Prefer depth as the primary foreground cue so low-texture standalone
        # objects (e.g. books on a floor) can still be segmented.
        foreground[valid_disparity] = 255
        # Keep colorful foreground even when disparity gets locally thin/noisy.
        foreground[valid_disparity & colorful] = 255
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1
        )

        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        candidates: list[dict[str, float | int | np.ndarray]] = []
        min_area = max(1, int(self.config.min_area * (scale**2)))

        for component_id in range(1, component_count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area or height <= 0:
                continue
            aspect_ratio = width / float(height)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            fill_ratio = area / float(max(1, width * height))
            if fill_ratio < self.config.min_fill_ratio:
                continue
            component_mask = component_labels == component_id
            edge_ratio = float(np.count_nonzero(edge_map[component_mask])) / float(area)
            if edge_ratio < self.config.min_edge_ratio:
                continue
            depth_values = disparity[component_mask]
            if depth_values.size == 0:
                continue
            mean_disparity = float(np.median(depth_values))
            # Require measurable depth separation from nearby background so we
            # rely more on stereo geometry than color alone.
            ring_radius = max(3, int(round(9 * scale)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1, ring_radius * 2 + 1)
            )
            expanded_mask = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1) > 0
            ring_mask = expanded_mask & (~component_mask)
            ring_depth_values = disparity[ring_mask & valid_disparity]
            if ring_depth_values.size > 20:
                background_disparity = float(np.median(ring_depth_values))
                depth_contrast = abs(mean_disparity - background_disparity)
                if depth_contrast < self.config.min_depth_contrast:
                    continue
            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "fill_ratio": fill_ratio,
                    "edge_ratio": edge_ratio,
                    "mean_disparity": mean_disparity,
                    "mask": component_mask,
                }
            )

        # Merge nearby color blobs when depth is continuous.
        parent = list(range(len(candidates)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                depth_gap = abs(float(a["mean_disparity"]) - float(b["mean_disparity"]))
                if depth_gap > self.config.depth_merge_threshold:
                    continue

                ax1, ay1 = int(a["x"]), int(a["y"])
                ax2, ay2 = ax1 + int(a["width"]), ay1 + int(a["height"])
                bx1, by1 = int(b["x"]), int(b["y"])
                bx2, by2 = bx1 + int(b["width"]), by1 + int(b["height"])
                horizontal_gap = max(0, max(bx1 - ax2, ax1 - bx2))
                vertical_gap = max(0, max(by1 - ay2, ay1 - by2))
                merge_gap = max(1, int(round(self.config.bbox_gap_merge_px * scale)))
                if max(horizontal_gap, vertical_gap) <= merge_gap:
                    union(i, j)

        grouped: dict[int, list[dict[str, float | int | np.ndarray]]] = {}
        for idx, candidate in enumerate(candidates):
            root = find(idx)
            grouped.setdefault(root, []).append(candidate)

        boxes: list[dict[str, float | int | str]] = []
        overlay = left.copy()
        mask_list: list[np.ndarray] = []
        for group in grouped.values():
            merged_mask = np.zeros_like(gray_left, dtype=np.uint8)
            for item in group:
                merged_mask[item["mask"]] = 255
            merged_mask = cv2.morphologyEx(
                merged_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1
            )
            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, self.config.epsilon_ratio * perimeter, True)
            x, y, width, height = cv2.boundingRect(contour)
            if height <= 0:
                continue
            aspect_ratio = width / float(height)
            fill_ratio = area / float(max(1, width * height))
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            if fill_ratio < self.config.min_fill_ratio:
                continue

            render_mask = merged_mask
            if scale < 1.0:
                render_mask = cv2.resize(
                    merged_mask,
                    (left.shape[1], left.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            color_layer = np.zeros_like(left, dtype=np.uint8)
            color_layer[render_mask > 0] = self.config.draw_color
            overlay = cv2.addWeighted(overlay, 1.0, color_layer, self.config.mask_alpha, 0)
            if scale < 1.0:
                approximation = np.round(approximation.astype(np.float32) / scale).astype(np.int32)
                x = int(round(x / scale))
                y = int(round(y / scale))
                width = int(round(width / scale))
                height = int(round(height / scale))
            cv2.drawContours(overlay, [approximation], -1, self.config.draw_color, 2)
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
                    "area": float(area),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "fill_ratio": round(fill_ratio, 3),
                    "mean_disparity": round(mean_disparity, 3),
                }
            )
            mask_list.append(render_mask)

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
