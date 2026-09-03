#!/usr/bin/env python3
"""Train an Ultralytics YOLO detection model on the prepared urchin v1 split.

Example cloud run:
    python train_yolo26m_urchins_v1.py --data dataset_review_and_split/split/data.yaml --device 0

Resume an interrupted run:
    python train_yolo26m_urchins_v1.py --resume training_runs/yolo26m_urchins_v1_seed_20260713/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_ROOT = DEFAULT_PROJECT_ROOT / "dataset_review_and_split" / "split"
DEFAULT_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26m on the prepared urchin v1 detection split.")
    parser.add_argument("--data", type=Path, default=DEFAULT_SPLIT_ROOT / "data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_ROOT / "training_runs", help="Ultralytics run output directory")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "models", help="Directory for copied/exported model artifacts")
    parser.add_argument("--model", default="yolo26m.pt", help="Primary pretrained model weights")
    parser.add_argument("--fallback-model", default="yolo11m.pt", help="Fallback weights if --model cannot be loaded")
    parser.add_argument("--no-fallback", action="store_true", help="Fail instead of falling back when --model cannot be loaded")
    parser.add_argument("--run-name", default="yolo26m_urchins_v1_seed_20260713", help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="DataLoader workers; 0 is safest in cloud containers with small /dev/shm")
    parser.add_argument("--shm-safe", action="store_true", help="Force settings that avoid DataLoader shared-memory bus errors")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers; omit for no freeze")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no-cos-lr", action="store_true", help="Disable cosine LR schedule")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Allow writing into an existing run folder")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset and runtime, then exit before training")
    parser.add_argument("--smoke-test", action="store_true", help="Run a one-epoch low-resolution training check before full training")
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt for an interrupted run")
    parser.add_argument("--skip-val", action="store_true", help="Skip val/test evaluation after training")
    parser.add_argument("--predict-samples", type=int, default=12, help="Number of test images for saved sample predictions; 0 disables")
    parser.add_argument("--predict-conf", type=float, default=0.25)
    parser.add_argument("--predict-max-det", type=int, default=300)
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_device(device_arg: str) -> int | str:
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_arg.lower() == "cpu":
        return "cpu"
    if "," not in device_arg:
        try:
            return int(device_arg)
        except ValueError:
            return device_arg
    return device_arg


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return completed.stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"


def apply_cloud_safety(args: argparse.Namespace) -> None:
    if args.shm_safe:
        args.workers = 0
        args.cache = False
        print(json.dumps({"shm_safe": True, "workers": args.workers, "cache": args.cache}, indent=2))


def check_runtime(device: int | str) -> None:
    runtime_info = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": device,
    }
    print(json.dumps(runtime_info, indent=2))

    if torch.cuda.is_available():
        print("nvidia-smi:")
        print(run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]))

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Use --device cpu or install GPU-enabled PyTorch.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_yaml_path(data_yaml_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_yaml_path.parent / path


def read_data_yaml(data_yaml_path: Path) -> dict[str, Any]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml_path}")
    with data_yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    names = config.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if not names:
        raise ValueError("data.yaml must define non-empty names")

    config["names"] = list(names)
    if int(config["nc"]) != len(config["names"]):
        raise ValueError(f"data.yaml nc={config['nc']} but names has {len(config['names'])} entries")
    for split_name in ["train", "val", "test"]:
        if split_name not in config:
            raise ValueError(f"data.yaml is missing required split: {split_name}")
    return config


def parse_label_file(label_path: Path, class_count: int) -> tuple[list[int], list[dict[str, Any]]]:
    class_ids: list[int] = []
    issues: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append({"label": str(label_path), "line": line_number, "issue": "expected_5_fields", "value": line})
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            issues.append({"label": str(label_path), "line": line_number, "issue": "non_numeric_value", "value": line})
            continue
        if class_id < 0 or class_id >= class_count:
            issues.append({"label": str(label_path), "line": line_number, "issue": "class_id_out_of_range", "value": class_id})
        if any(coord < 0 or coord > 1 for coord in coords):
            issues.append({"label": str(label_path), "line": line_number, "issue": "bbox_coord_out_of_range", "value": coords})
        class_ids.append(class_id)
    return class_ids, issues


def validate_split(data_yaml_path: Path) -> dict[str, Any]:
    data_config = read_data_yaml(data_yaml_path)
    class_names = data_config["names"]
    class_count = int(data_config["nc"])
    split_rows: list[dict[str, Any]] = []
    pair_issues: list[dict[str, Any]] = []
    label_issues: list[dict[str, Any]] = []
    class_counter: Counter[tuple[str, int]] = Counter()
    image_class_counter: Counter[tuple[str, int]] = Counter()

    for split_name in ["train", "val", "test"]:
        image_dir = resolve_yaml_path(data_yaml_path, data_config[split_name])
        label_dir = data_yaml_path.parent / "labels" / split_name
        if not image_dir.exists():
            pair_issues.append({"split": split_name, "issue": "missing_image_dir", "path": str(image_dir)})
            continue
        if not label_dir.exists():
            pair_issues.append({"split": split_name, "issue": "missing_label_dir", "path": str(label_dir)})
            continue

        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        label_paths = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        if missing_labels:
            pair_issues.append({"split": split_name, "issue": "missing_labels", "count": len(missing_labels), "examples": missing_labels[:5]})
        if orphan_labels:
            pair_issues.append({"split": split_name, "issue": "orphan_labels", "count": len(orphan_labels), "examples": orphan_labels[:5]})

        empty_label_files = 0
        object_count = 0
        for label_path in label_paths:
            class_ids, issues = parse_label_file(label_path, class_count)
            label_issues.extend(issues)
            if not class_ids:
                empty_label_files += 1
            object_count += len(class_ids)
            class_counter.update((split_name, class_id) for class_id in class_ids if 0 <= class_id < class_count)
            for class_id in set(class_ids):
                if 0 <= class_id < class_count:
                    image_class_counter.update([(split_name, class_id)])

        split_rows.append({
            "split": split_name,
            "images": len(image_paths),
            "labels": len(label_paths),
            "objects": object_count,
            "empty_label_files": empty_label_files,
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
        })

    split_df = pd.DataFrame(split_rows)
    class_df = pd.DataFrame([
        {
            "split": split_name,
            "class_id": class_id,
            "class_name": class_names[class_id],
            "annotations_boxes": class_counter[(split_name, class_id)],
            "images_with_class": image_class_counter[(split_name, class_id)],
        }
        for split_name in ["train", "val", "test"]
        for class_id in range(class_count)
    ])

    print("split validation:")
    print(split_df.to_string(index=False))
    print("\nannotations/boxes by class:")
    print(class_df.pivot(index="class_name", columns="split", values="annotations_boxes").fillna(0).astype(int).to_string())
    print("\nimages with class:")
    print(class_df.pivot(index="class_name", columns="split", values="images_with_class").fillna(0).astype(int).to_string())

    if pair_issues or label_issues:
        if pair_issues:
            print("pair issues:")
            print(pd.DataFrame(pair_issues).to_string(index=False))
        if label_issues:
            print("label issues:")
            print(pd.DataFrame(label_issues).head(50).to_string(index=False))
        raise ValueError(f"Dataset validation failed with {len(pair_issues)} pair issues and {len(label_issues)} label issues.")

    return {"data_config": data_config, "class_names": class_names, "split_df": split_df, "class_df": class_df}


def load_yolo_model(primary_model: str, fallback_model: str | None) -> tuple[YOLO, str]:
    candidates = [primary_model]
    if fallback_model and fallback_model not in candidates:
        candidates.append(fallback_model)

    errors: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        try:
            model = YOLO(candidate)
            if index > 0:
                print(json.dumps({"model_fallback_used": candidate, "primary_model_error": errors.get(primary_model)}, indent=2))
            return model, candidate
        except Exception as error:  # noqa: BLE001
            errors[candidate] = str(error)
            if index == 0 and fallback_model is None:
                raise RuntimeError(f"Could not load {primary_model}. Try upgrading ultralytics or choose another --model.") from error
    raise RuntimeError(f"Could not load any model candidate: {errors}")


def run_smoke_test(args: argparse.Namespace, data_yaml_path: Path, device: int | str, fallback_model: str | None) -> None:
    smoke_model, smoke_model_name = load_yolo_model(args.model, fallback_model)
    smoke_results = smoke_model.train(
        data=str(data_yaml_path),
        epochs=1,
        imgsz=min(args.imgsz, 640),
        batch=min(args.batch, 2),
        device=device,
        workers=min(args.workers, 2),
        project=str(args.project),
        name=f"{args.run_name}_smoke",
        exist_ok=True,
        plots=False,
    )
    print(json.dumps({"smoke_test_save_dir": str(smoke_results.save_dir), "model": smoke_model_name}, indent=2))


def train(args: argparse.Namespace, data_yaml_path: Path, device: int | str, fallback_model: str | None) -> tuple[Path, Path, Path]:
    if args.resume:
        resume_path = resolve_path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        resume_model = YOLO(str(resume_path))
        resume_results = resume_model.train(resume=True, workers=args.workers, cache=args.cache, device=device)
        run_dir = Path(resume_results.save_dir)
        return run_dir, run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"

    model, selected_model_name = load_yolo_model(args.model, fallback_model)
    train_args: dict[str, Any] = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "cos_lr": not args.no_cos_lr,
        "patience": args.patience,
        "save_period": args.save_period,
        "seed": args.seed,
        "deterministic": True,
        "amp": not args.no_amp and torch.cuda.is_available(),
        "cache": args.cache,
        "close_mosaic": args.close_mosaic,
        "project": str(args.project),
        "name": args.run_name,
        "exist_ok": args.exist_ok,
        "plots": True,
        "verbose": True,
    }
    if args.freeze is not None:
        train_args["freeze"] = args.freeze

    print(json.dumps({"selected_model_name": selected_model_name, "train_args": train_args}, indent=2))
    train_results = model.train(**train_args)
    run_dir = Path(train_results.save_dir)
    return run_dir, run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"


def evaluate(weights_path: Path, data_yaml_path: Path, args: argparse.Namespace, device: int | str) -> None:
    model = YOLO(str(weights_path))
    val_metrics = model.val(data=str(data_yaml_path), split="val", imgsz=args.imgsz, batch=args.batch, device=device, plots=True)
    test_metrics = model.val(data=str(data_yaml_path), split="test", imgsz=args.imgsz, batch=args.batch, device=device, plots=True)
    print(json.dumps({"weights": str(weights_path), "val_metrics": str(val_metrics), "test_metrics": str(test_metrics)}, indent=2))


def predict_samples(weights_path: Path, data_yaml_path: Path, args: argparse.Namespace, device: int | str) -> None:
    if args.predict_samples <= 0:
        return
    test_images_dir = data_yaml_path.parent / "images" / "test"
    sample_images = sorted(path for path in test_images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)[: args.predict_samples]
    if not sample_images:
        print(json.dumps({"prediction_skipped": f"No test images found in {test_images_dir}"}, indent=2))
        return

    model = YOLO(str(weights_path))
    prediction_results = model.predict(
        source=[str(path) for path in sample_images],
        imgsz=args.imgsz,
        conf=args.predict_conf,
        max_det=args.predict_max_det,
        device=device,
        save=True,
        save_txt=True,
        project=str(args.project),
        name=f"{args.run_name}_test_predictions",
        exist_ok=True,
    )
    print(json.dumps({"prediction_dir": str(prediction_results[0].save_dir), "sample_count": len(sample_images)}, indent=2))


def copy_and_export(weights_path: Path, args: argparse.Namespace) -> None:
    args.models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    copied_best_path = args.models_dir / f"{args.run_name}_best_{timestamp}.pt"
    shutil.copy2(weights_path, copied_best_path)
    print(json.dumps({"copied_best_weights": str(copied_best_path)}, indent=2))

    if args.skip_export:
        print(json.dumps({"onnx_export": "skipped"}, indent=2))
        return

    try:
        model = YOLO(str(weights_path))
        exported_path = model.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=False)
        print(json.dumps({"onnx_export": str(exported_path)}, indent=2))
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"onnx_export_error": str(error)}, indent=2))


def main() -> None:
    args = parse_args()
    apply_cloud_safety(args)
    args.data = resolve_path(args.data)
    args.project = resolve_path(args.project)
    args.models_dir = resolve_path(args.models_dir)
    args.project.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    fallback_model = None if args.no_fallback else args.fallback_model

    check_runtime(device)
    set_seed(args.seed)
    validation = validate_split(args.data)
    print(json.dumps({"data_yaml": str(args.data), "nc": len(validation["class_names"]), "names": validation["class_names"], "validation": "passed"}, indent=2))

    if args.validate_only:
        print(json.dumps({"status": "validate_only_complete"}, indent=2))
        return

    if args.smoke_test:
        run_smoke_test(args, args.data, device, fallback_model)

    run_dir, best_weights, last_weights = train(args, args.data, device, fallback_model)
    weights_for_eval = best_weights if best_weights.exists() else last_weights
    if not weights_for_eval.exists():
        raise FileNotFoundError(f"No trained weights found in {run_dir / 'weights'}")

    print(json.dumps({"run_dir": str(run_dir), "best_weights": str(best_weights), "last_weights": str(last_weights)}, indent=2))

    if not args.skip_val:
        evaluate(weights_for_eval, args.data, args, device)
    predict_samples(weights_for_eval, args.data, args, device)
    copy_and_export(weights_for_eval, args)


if __name__ == "__main__":
    main()
