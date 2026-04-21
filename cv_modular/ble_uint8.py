from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class BleUint8ServerConfig:
    adapter_address: str = "B8:27:EB:00:00:01"
    local_name: str = "FingerCountPi"
    service_uuid: str = "12345678-1234-5678-1234-56789abcdef0"
    characteristic_uuid: str = "12345678-1234-5678-1234-56789abcdef1"


class BleUint8Server:
    """BLE GATT server that exposes one uint8 characteristic for nRF Connect."""

    def __init__(self, config: BleUint8ServerConfig) -> None:
        try:
            from bluezero import peripheral
        except Exception as exc:
            raise RuntimeError(
                "BLE uint8 support requires bluezero. Install it with: pip install bluezero"
            ) from exc

        self._config = config
        self._value: list[int] = [0]
        self._notifications_enabled = False
        self._peripheral = peripheral.Peripheral(adapter_addr=config.adapter_address, local_name=config.local_name)
        self._peripheral.add_service(srv_id=1, uuid=config.service_uuid, primary=True)
        self._peripheral.add_characteristic(
            srv_id=1,
            chr_id=1,
            uuid=config.characteristic_uuid,
            value=self._value,
            notifying=False,
            flags=["read", "notify"],
            read_callback=self._read_callback,
            write_callback=None,
            notify_callback=self._notify_callback,
        )

        self._thread = threading.Thread(target=self._peripheral.publish, daemon=True)
        self._thread.start()

    def _read_callback(self, _options=None) -> list[int]:
        return self._value

    def _notify_callback(self, notifying: bool, _characteristic) -> None:
        self._notifications_enabled = notifying

    def set_value(self, value: int) -> None:
        clamped = max(0, min(255, int(value)))
        self._value = [clamped]
        self._peripheral.update_characteristic_value(1, 1, self._value)

    def notify(self) -> None:
        if not self._notifications_enabled:
            return
        self._peripheral.notify(1, 1)

    def close(self) -> None:
        try:
            self._peripheral.stop()
        except Exception:
            pass
