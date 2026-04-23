# Run automatically on Raspberry Pi boot (no monitor needed)

This project now supports **headless mode** and includes a **systemd installer** so it starts as soon as the Pi boots.

## 1) One-time setup on the Pi

From the repo root:

```bash
chmod +x deploy/systemd/install_autostart_service.sh
./deploy/systemd/install_autostart_service.sh
```

What this does:
- creates `/etc/systemd/system/retail-inventory-manager.service`
- enables it at boot (`systemctl enable`)
- starts/restarts it immediately

## 2) Verify it is running

```bash
sudo systemctl status retail-inventory-manager.service
journalctl -u retail-inventory-manager.service -f
```

## 3) Change runtime options

The installer accepts environment overrides:

```bash
RUN_ARGS="--headless --button-controlled --button-gpio-pin 17" ./deploy/systemd/install_autostart_service.sh
```

Optional overrides:
- `PYTHON_BIN=/custom/venv/bin/python`
- `TARGET_USER=pi`
- `TARGET_GROUP=pi`

## 4) Disable autostart

```bash
sudo systemctl disable --now retail-inventory-manager.service
```

## 5) Headless mode details

Use `--headless` to run without opening an OpenCV window. This is required for monitor-free boot startup.
