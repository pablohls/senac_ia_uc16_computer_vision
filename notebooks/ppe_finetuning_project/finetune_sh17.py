from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finetuning YOLOv8 on SH17 PPE dataset"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/images/sh17_dataset/data.yaml"),
        help="Path to SH17 data.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("notebooks/yolov8n.pt"),
        help="Path to base model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", type=str, default="sh17_yolov8n_finetune")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only on an existing model checkpoint",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"model checkpoint not found: {args.model}")

    model = YOLO(str(args.model))

    if args.validate_only:
        metrics = model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device)
        print("Validation finished.")
        print(metrics)
        return

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        pretrained=True,
    )
    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
