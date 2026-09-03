#!/usr/bin/env python3
"""Train YOLO classification model on the prepared bleaching dataset split.

Run this from a conda environment with torch + ultralytics installed.
Example:
    conda activate coral-train
    python train_yolo_bleaching.py --dataset ~/bleaching_classifier_training/dataset_split
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO classifier for coral bleaching dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / "bleaching_classifier_training" / "dataset_split",
        help="Path containing train/val/test class folders",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.home() / "bleaching_classifier_training" / "training_logs",
        help="Directory where Ultralytics writes run outputs",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path.home() / "bleaching_classifier_training" / "models",
        help="Directory to store copied best model artifacts",
    )
    parser.add_argument("--model", default="yolo11m-cls.pt", help="Base model weights")
    parser.add_argument("--run-name", default="yolo11m_cls_bleaching_seeded_split", help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running when CUDA is unavailable")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    return parser.parse_args()


def check_runtime(cuda_required: bool) -> None:
    print({
        "python_executable": __import__("sys").executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    try:
        smi = subprocess.run(
            ["/var/lib/nvidia/bin/nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("nvidia-smi:")
        print(smi.stdout.strip())
    except Exception as error:  # noqa: BLE001
        print({"nvidia_smi_check_error": str(error)})

    if cuda_required and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in torch. Activate a GPU-enabled runtime or pass --allow-cpu to continue on CPU."
        )


def validate_dataset(dataset_path: Path) -> list[str]:
    required_splits = ["train", "val"]
    missing_splits = [split for split in required_splits if not (dataset_path / split).exists()]
    if missing_splits:
        raise FileNotFoundError(f"Missing dataset split folders: {missing_splits}")

    classes = sorted(path.name for path in (dataset_path / "train").iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subfolders found under {(dataset_path / 'train')}")
    return classes


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    project_path = args.project.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()

    project_path.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    check_runtime(cuda_required=not args.allow_cpu)
    classes = validate_dataset(dataset_path)
    set_seed(args.seed)

    print(
        {
            "dataset_path": str(dataset_path),
            "available_classes": classes,
            "seed": args.seed,
            "device": 0 if torch.cuda.is_available() else "cpu",
            "project": str(project_path),
            "models_dir": str(models_dir),
        }
    )

    model = YOLO(args.model)

    results = model.train(
        data=str(dataset_path),
        device=0 if torch.cuda.is_available() else "cpu",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=torch.cuda.is_available(),
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.01,
        weight_decay=0.001,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        deterministic=True,
        project=str(project_path),
        name=args.run_name,
    )

    best_weights_path = Path(results.save_dir) / "weights" / "best.pt"
    final_model_path = models_dir / "yolo11m-cls-bleaching-seeded-split.pt"

    if best_weights_path.exists():
        final_model_path.write_bytes(best_weights_path.read_bytes())
        print({"best_model_copied_to": str(final_model_path)})
    else:
        print({"best_weights_not_found": str(best_weights_path)})

    if not args.skip_export:
        try:
            model.export(format="onnx")
            print("ONNX export succeeded")
        except Exception as error:  # noqa: BLE001
            print({"onnx_export_error": str(error)})

    metrics = model.val(data=str(dataset_path), device=0 if torch.cuda.is_available() else "cpu")
    print(metrics)


if __name__ == "__main__":
    main()
