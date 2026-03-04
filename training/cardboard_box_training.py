#!/usr/bin/env python3
"""Train a cardboard-box detector using a Roboflow dataset and YOLOv8.

This script is additive to the existing CV pipeline and does not modify runtime code.
It can:
1) Download/export a Roboflow dataset in YOLOv8 format.
2) Train a YOLOv8 model.
3) Run validation and a sample prediction pass.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cardboard box detector from Roboflow.")
    parser.add_argument("--api-key", default=None, help="Roboflow API key. If omitted, uses ROBOFLOW_API_KEY env var.")
    parser.add_argument("--workspace", default="dataset-t7hz7", help="Roboflow workspace slug.")
    parser.add_argument("--project", default="cardboard-eupc8", help="Roboflow project slug.")
    parser.add_argument("--version", type=int, required=True, help="Roboflow dataset version number.")
    parser.add_argument("--format", default="yolov8", help="Dataset export format (default: yolov8).")
    parser.add_argument("--dataset-dir", default="datasets", help="Directory where dataset will be downloaded.")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download and only train.")
    parser.add_argument("--data-yaml", default=None, help="Path to data.yaml. Required when --skip-download is used.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--project-name", default="cardboard_training", help="Ultralytics project output name.")
    parser.add_argument("--run-name", default="yolov8_cardboard", help="Ultralytics run name.")
    parser.add_argument("--predict-split", default="test", choices=["train", "valid", "test"], help="Split for sample prediction output.")
    return parser.parse_args()


def resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.skip_download:
        if not args.data_yaml:
            raise ValueError("--data-yaml is required when --skip-download is set.")
        data_yaml = Path(args.data_yaml).expanduser().resolve()
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
        return data_yaml

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set ROBOFLOW_API_KEY to download from Roboflow.")

    from roboflow import Roboflow

    download_root = Path(args.dataset_dir).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format, location=str(download_root))

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Expected data.yaml at {data_yaml}, but it was not found.")

    return data_yaml


def main() -> None:
    args = parse_args()
    data_yaml = resolve_data_yaml(args)

    from ultralytics import YOLO

    model = YOLO(args.model)

    train_results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project_name,
        name=args.run_name,
        exist_ok=True,
    )

    model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)

    split_folder = data_yaml.parent / args.predict_split / "images"
    if split_folder.exists():
        model.predict(
            source=str(split_folder),
            imgsz=args.imgsz,
            conf=0.25,
            save=True,
            project=args.project_name,
            name=f"{args.run_name}_{args.predict_split}_predictions",
            exist_ok=True,
        )

    print("Training complete.")
    print(f"Run directory: {Path(train_results.save_dir).resolve()}")
    print("Best weights are typically at: <run_dir>/weights/best.pt")


if __name__ == "__main__":
    main()
