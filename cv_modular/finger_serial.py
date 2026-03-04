from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class FingerSerialSenderConfig:
    port: str
    baud: int = 115200
    timeout_s: float = 1.0
    min_interval_s: float = 0.05
    message_prefix: str = "FINGERS"


class FingerSerialSender:
    """Sends finger-count updates to a serial device (e.g. Arduino)."""

    def __init__(self, config: FingerSerialSenderConfig) -> None:
        self.config = config
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for --serial-port support. Install with: pip install pyserial"
            ) from exc

        self.serial = serial.Serial(config.port, config.baud, timeout=config.timeout_s)
        # Let Arduino-style boards reboot after serial connection.
        time.sleep(2.0)

        self._last_count: int | None = None
        self._last_send_time = 0.0

    def send_finger_count(self, finger_count: int) -> None:
        now = time.monotonic()
        if now - self._last_send_time < self.config.min_interval_s:
            return
        if finger_count == self._last_count:
            return

        msg = f"{self.config.message_prefix}:{finger_count}\n"
        self.serial.write(msg.encode("utf-8"))
        self._last_count = finger_count
        self._last_send_time = now

    def close(self) -> None:
        self.serial.close()
