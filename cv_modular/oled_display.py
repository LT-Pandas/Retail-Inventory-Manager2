from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OledDisplayConfig:
    """Configuration for SPI OLED output using luma.oled."""

    spi_port: int = 0
    spi_device: int = 0
    gpio_dc: int = 25
    gpio_rst: int = 24
    driver: str = "auto"  # one of: auto, sh1107, ssd1309, ssd1327


class OledCountDisplay:
    """Render the current count as a single number on an SPI OLED display."""

    _AUTO_DRIVERS = ["ssd1309", "sh1107", "ssd1327"]

    def __init__(self, config: OledDisplayConfig | None = None) -> None:
        self.config = config or OledDisplayConfig()

        try:
            from luma.core.interface.serial import spi
            from luma.core.render import canvas
            from luma.oled import device as oled_devices
        except Exception as exc:  # pragma: no cover - import guard for optional dependency
            raise RuntimeError(
                "OLED display support requires luma.oled. Install it with: "
                "pip install luma.oled luma.core"
            ) from exc

        self._canvas = canvas
        self._device = None
        serial = spi(
            port=self.config.spi_port,
            device=self.config.spi_device,
            gpio_DC=self.config.gpio_dc,
            gpio_RST=self.config.gpio_rst,
        )

        if self.config.driver == "auto":
            drivers_to_try = self._AUTO_DRIVERS
        else:
            drivers_to_try = [self.config.driver]

        init_errors: list[str] = []
        for driver_name in drivers_to_try:
            driver = getattr(oled_devices, driver_name, None)
            if driver is None:
                init_errors.append(f"{driver_name}: not available in installed luma.oled version")
                continue

            try:
                self._device = driver(serial)
                self._driver_name = driver_name
                break
            except Exception as exc:
                init_errors.append(f"{driver_name}: {exc}")

        if self._device is None:
            joined_errors = "; ".join(init_errors)
            raise RuntimeError(
                "Unable to initialize OLED display. Tried drivers "
                f"{', '.join(drivers_to_try)}. Errors: {joined_errors}"
            )

        self._last_count: int | None = None

    def render_message(self, message: str, position: tuple[int, int] = (0, 0)) -> None:
        with self._canvas(self._device) as draw:
            draw.rectangle(self._device.bounding_box, outline=0, fill=0)
            draw.text(position, message, fill=255)

    def render_count(self, count: int) -> None:
        if count == self._last_count:
            return

        self._last_count = count
        text = str(count)

        with self._canvas(self._device) as draw:
            draw.rectangle(self._device.bounding_box, outline=0, fill=0)
            draw.text((8, 8), text, fill=255)

    def close(self) -> None:
        if self._device is None:
            return

        with self._canvas(self._device) as draw:
            draw.rectangle(self._device.bounding_box, outline=0, fill=0)

    @property
    def active_driver(self) -> str:
        return self._driver_name
