#!/usr/bin/env python3
"""Train an Ultralytics YOLO detector on the local license-plate dataset."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a local pretrained YOLO .pt checkpoint, e.g. weights/yolo11n.pt. "
             "Pass a local file to avoid automatic downloads.",
    )
    parser.add_argument("--data", default="data.yaml", help="Dataset YAML path (default: data.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help='Device: "0", "cpu", or e.g. "0,1"')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="license_plate_detector")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk if supported")
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Download/prepare the checkpoint separately and pass its local path with --model."
        )
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = data_path.resolve().parent / dataset_root

    for split in ("train", "val", "test"):
        image_dir = dataset_root / config[split]
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} image directory: {image_dir}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        plots=True,
        save=True,
    )

    # Evaluate the best checkpoint on the held-out test split after training.
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.is_file():
        best_model = YOLO(str(best_path))
        metrics = best_model.val(
            data=str(data_path.resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
        )
        print(f"Test metrics: {metrics.results_dict}")
        print(f"Best checkpoint: {best_path.resolve()}")
    else:
        print(f"Training completed, but best checkpoint was not found at: {best_path}")


if __name__ == "__main__":
    main()
