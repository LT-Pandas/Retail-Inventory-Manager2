from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
from cv_modular import CVPipeline, run_webcam_loop
from cv_modular.ble_uint8 import BleUint8Server, BleUint8ServerConfig
from cv_modular.oled_display import OledCountDisplay, OledDisplayConfig
from cv_modular.processors import (
    BoxDetectorConfig,
    BoxDetectorProcessor,
)


def _extract_object_total(results) -> int | None:
    for result in results:
        if result.name == "box_detector":
            return result.data.get("count")
    return None


def _build_live_count_graph_frame(
    count_history: deque[int],
    max_history: int,
    current_count: int,
    width: int = 760,
    height: int = 360,
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)

    panel_margin = 16
    panel_x = panel_margin
    panel_y = panel_margin
    panel_width = width - (panel_margin * 2)
    panel_height = height - (panel_margin * 2)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (45, 45, 45), 1)

    title_y = panel_y + 28
    cv2.putText(
        frame,
        "Detection Stability (Live Object Count)",
        (panel_x + 16, title_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.68,
        (255, 240, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Current: {current_count}",
        (panel_x + panel_width - 180, title_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (100, 235, 255),
        1,
        cv2.LINE_AA,
    )

    graph_left = panel_x + 18
    graph_top = panel_y + 48
    graph_right = panel_x + panel_width - 16
    graph_bottom = panel_y + panel_height - 18
    graph_width = graph_right - graph_left
    graph_height = graph_bottom - graph_top

    for i in range(4):
        y = graph_top + int((graph_height / 3) * i)
        cv2.line(frame, (graph_left, y), (graph_right, y), (64, 64, 64), 1, cv2.LINE_AA)

    for i in range(5):
        x = graph_left + int((graph_width / 4) * i)
        cv2.line(frame, (x, graph_top), (x, graph_bottom), (45, 45, 45), 1, cv2.LINE_AA)

    history_values = list(count_history)
    if len(history_values) < 2:
        cv2.putText(
            frame,
            "Collecting samples...",
            (graph_left + 8, graph_top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        return frame

    max_value = max(1, max(history_values))
    points: list[tuple[int, int]] = []
    for idx, value in enumerate(history_values):
        x = graph_left + int(idx * (graph_width / max(1, max_history - 1)))
        normalized = value / max_value
        y = graph_bottom - int(normalized * graph_height)
        points.append((x, y))

    if len(points) >= 2:
        area_points = [(points[0][0], graph_bottom), *points, (points[-1][0], graph_bottom)]
        area = np.array(area_points, dtype=np.int32).reshape((-1, 1, 2))
        area_overlay = frame.copy()
        cv2.fillPoly(area_overlay, [area], (245, 170, 55))
        cv2.addWeighted(area_overlay, 0.20, frame, 0.80, 0, frame)

        line_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [line_points], False, (80, 220, 255), 3, cv2.LINE_AA)

        cv2.circle(frame, points[-1], 5, (85, 255, 120), -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"0",
        (graph_left - 12, graph_bottom + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{max_value}",
        (graph_left - 16, graph_top + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object detection CV demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--fallback-camera-indexes",
        type=int,
        nargs="*",
        default=[1],
        help="Additional camera indexes; first unique value is stitched with --camera-index (default: 1).",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=2500,
        help="Minimum contour area for an object candidate.",
    )
    parser.add_argument(
        "--object-epsilon-ratio",
        type=float,
        default=0.04,
        help="Contour simplification epsilon ratio used in polygon approximation.",
    )
    parser.add_argument(
        "--object-min-aspect-ratio",
        type=float,
        default=0.1,
        help="Minimum bounding box aspect ratio for contour-based object detection.",
    )
    parser.add_argument(
        "--object-max-aspect-ratio",
        type=float,
        default=10.0,
        help="Maximum bounding box aspect ratio for contour-based object detection.",
    )
    parser.add_argument(
        "--stereo-calibration-file",
        type=str,
        default="stereo_calibration.npz",
        help="Optional .npz file containing rectification maps for left/right cameras.",
    )
    parser.add_argument(
        "--stereo-disparity-threshold",
        type=float,
        default=1.0,
        help="Minimum disparity considered foreground in stereo depth mask.",
    )
    parser.add_argument(
        "--depth-merge-threshold",
        type=float,
        default=4.0,
        help="Maximum disparity delta to merge nearby color blobs into one object.",
    )
    parser.add_argument(
        "--object-min-edge-ratio",
        type=float,
        default=0.012,
        help="Minimum edge density inside a candidate; helps reject soft shadows.",
    )
    parser.add_argument(
        "--stereo-processing-scale",
        type=float,
        default=0.6,
        help="Downscale factor used internally for stereo matching to improve FPS (0.25-1.0).",
    )
    parser.add_argument(
        "--stereo-bbox-merge-gap",
        type=int,
        default=42,
        help="Maximum pixel gap between components eligible for depth-based merge.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.28,
        help="Opacity of rendered object masks on the left camera frame.",
    )
    parser.add_argument("--camera-width", type=int, default=1920, help="Camera capture width in pixels.")
    parser.add_argument("--camera-height", type=int, default=1080, help="Camera capture height in pixels.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Target camera FPS.")
    parser.add_argument(
        "--camera-brightness",
        type=float,
        default=0.08,
        help="Camera brightness control; small positive values can lift shadows.",
    )
    parser.add_argument("--camera-contrast", type=float, default=1.2, help="Camera contrast control.")
    parser.add_argument("--camera-saturation", type=float, default=1.15, help="Camera saturation control.")
    parser.add_argument("--camera-sharpness", type=float, default=1.0, help="Camera sharpness control.")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.6,
        help=(
            "Scale factor used for the on-screen preview window. "
            "Use values < 1.0 to shrink the stitched camera frame so it fits on one screen."
        ),
    )
    parser.add_argument(
        "--no-camera-autofocus",
        action="store_true",
        help="Disable camera autofocus if your module supports focus controls.",
    )
    parser.add_argument(
        "--camera-lens-position",
        type=float,
        default=None,
        help="Optional manual lens position (requires --no-camera-autofocus and supported camera hardware).",
    )
    parser.add_argument(
        "--ble-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable BLE uint8 GATT server for nRF Connect phone app (default: disabled).",
    )
    parser.add_argument(
        "--ble-adapter-address",
        type=str,
        default="B8:27:EB:00:00:01",
        help="Bluetooth adapter MAC address used by BlueZ.",
    )
    parser.add_argument(
        "--ble-local-name",
        type=str,
        default="FingerCountPi",
        help="Advertised BLE local name shown in nRF Connect.",
    )
    parser.add_argument(
        "--ble-service-uuid",
        type=str,
        default="12345678-1234-5678-1234-56789abcdef0",
        help="BLE service UUID for object count.",
    )
    parser.add_argument(
        "--ble-characteristic-uuid",
        type=str,
        default="12345678-1234-5678-1234-56789abcdef1",
        help="BLE characteristic UUID for uint8 object count.",
    )
    parser.add_argument(
        "--oled-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable OLED SPI output for object count (default: enabled). Use --no-oled-enabled to disable.",
    )
    parser.add_argument(
        "--oled-driver",
        choices=["auto", "sh1107", "ssd1309", "ssd1327"],
        default="auto",
        help="OLED driver to use. Default auto tries ssd1309, sh1107, then ssd1327.",
    )
    parser.add_argument("--oled-spi-port", type=int, default=0, help="OLED SPI port index.")
    parser.add_argument("--oled-spi-device", type=int, default=0, help="OLED SPI chip-select device index.")
    parser.add_argument("--oled-gpio-dc", type=int, default=25, help="OLED DC GPIO pin.")
    parser.add_argument("--oled-gpio-rst", type=int, default=24, help="OLED reset GPIO pin.")
    parser.add_argument(
        "--oled-test-message",
        type=str,
        default="RPI5 OLED OK",
        help="Startup test text shown once when OLED is enabled.",
    )
    parser.add_argument(
        "--oled-test-seconds",
        type=float,
        default=2.0,
        help="How long to display the startup OLED test text.",
    )
    parser.add_argument(
        "--button-controlled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require a physical button to start CV processing. "
            "A press substitutes clicking the triangle run button."
        ),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run without opening the OpenCV preview window. "
            "Useful for boot-time systemd startup on Raspberry Pi without a monitor."
        ),
    )
    parser.add_argument(
        "--stability-graph-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show a dedicated OpenCV window with a live count trend graph. "
            "Useful for testing detection stability over time."
        ),
    )
    parser.add_argument(
        "--button-gpio-pin",
        type=int,
        default=17,
        help=("BCM GPIO pin used for button input when --button-controlled is enabled. "
             "Important: BCM 17 is physical pin 11 (NOT physical pin 17)."),
    )
    parser.add_argument(
        "--button-hold-time",
        type=float,
        default=0.05,
        help="Minimum button hold time in seconds to register a press.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    processors = [
        BoxDetectorProcessor(
            BoxDetectorConfig(
                min_area=args.min_object_area,
                epsilon_ratio=args.object_epsilon_ratio,
                min_aspect_ratio=args.object_min_aspect_ratio,
                max_aspect_ratio=args.object_max_aspect_ratio,
                disparity_foreground_threshold=args.stereo_disparity_threshold,
                depth_merge_threshold=args.depth_merge_threshold,
                min_edge_ratio=args.object_min_edge_ratio,
                bbox_gap_merge_px=args.stereo_bbox_merge_gap,
                mask_alpha=args.mask_alpha,
                processing_scale=args.stereo_processing_scale,
                calibration_file=args.stereo_calibration_file,
            )
        )
    ]
    pipeline = CVPipeline(processors)

    button = None
    if args.button_controlled:
        try:
            from gpiozero import Button
        except Exception as exc:
            raise RuntimeError(
                "Button control requires gpiozero. Install it with `pip install gpiozero`."
            ) from exc

        button = Button(args.button_gpio_pin, pull_up=True, hold_time=args.button_hold_time, bounce_time=0.1)

        print(
            "Button control enabled on BCM GPIO "
            f"{args.button_gpio_pin}. BCM numbering is used by this app. "
            "If using BCM 17, wire to physical pin 11 (NOT physical pin 17). "
            "Wire the other button lead to a GND pin (for example physical pin 14). "
            "Press once to start (equivalent to pressing the triangle run button)."
        )

    ble_server = None
    last_total: int | None = None
    max_graph_history = 100
    count_history: deque[int] = deque(maxlen=max_graph_history)
    graph_window_name = "Detection Stability Graph"
    if args.ble_enabled:
        ble_server = BleUint8Server(
            BleUint8ServerConfig(
                adapter_address=args.ble_adapter_address,
                local_name=args.ble_local_name,
                service_uuid=args.ble_service_uuid,
                characteristic_uuid=args.ble_characteristic_uuid,
            )
        )
        print("BLE uint8 server enabled.")
        if args.headless:
            print("In nRF Connect: connect and enable notifications (headless mode has no keyboard trigger).")
        else:
            print("In nRF Connect: connect, enable notifications, then press 'o' in the CV window.")

    oled_display = None
    if args.oled_enabled:
        oled_display = OledCountDisplay(
            OledDisplayConfig(
                spi_port=args.oled_spi_port,
                spi_device=args.oled_spi_device,
                gpio_dc=args.oled_gpio_dc,
                gpio_rst=args.oled_gpio_rst,
                driver=args.oled_driver,
            )
        )
        print(f"OLED display enabled using driver: {oled_display.active_driver}")
        if args.oled_test_message:
            oled_display.render_message(args.oled_test_message)
            print(f"OLED test message: {args.oled_test_message!r}")
            if args.oled_test_seconds > 0:
                time.sleep(args.oled_test_seconds)

    def on_output(output) -> None:
        nonlocal last_total
        total = _extract_object_total(output.results)
        if total is None:
            return
        last_total = total
        count_history.append(total)
        if args.stability_graph_enabled and not args.headless:
            graph_frame = _build_live_count_graph_frame(
                count_history=count_history,
                max_history=max_graph_history,
                current_count=total,
            )
            cv2.imshow(graph_window_name, graph_frame)

        if oled_display is not None:
            oled_display.render_count(total)

        if ble_server is not None:
            ble_server.set_value(total)

    def on_key(key: int) -> bool:
        if key == ord("o") and ble_server is not None:
            if last_total is None:
                print("No object count available yet; nothing sent over BLE.")
                return False
            ble_server.set_value(last_total)
            ble_server.notify()
            print(f"BLE notify sent (uint8): {last_total}")
        return False

    def run_cv_once() -> None:
        run_webcam_loop(
            pipeline,
            camera_index=args.camera_index,
            fallback_camera_indexes=args.fallback_camera_indexes,
            on_output=on_output,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            camera_brightness=args.camera_brightness,
            camera_contrast=args.camera_contrast,
            camera_saturation=args.camera_saturation,
            camera_sharpness=args.camera_sharpness,
            camera_autofocus=not args.no_camera_autofocus,
            camera_lens_position=args.camera_lens_position,
            display_scale=args.display_scale,
            on_key=on_key,
            show_window=not args.headless,
        )

    try:
        if args.headless:
            print("Headless mode enabled: no OpenCV window will be shown.")
        if args.button_controlled:
            while True:
                print("Waiting for button press to start...")
                button.wait_for_press()
                print("Starting CV loop after button press.")
                run_cv_once()
                print("CV loop exited. Press the button to run again, or Ctrl+C to quit.")
        else:
            run_cv_once()
    finally:
        if ble_server is not None:
            ble_server.close()
        if oled_display is not None:
            oled_display.close()
        if button is not None:
            button.close()


if __name__ == "__main__":
    main()
