# YOLO Training Template

This template is intended as a starting point for a new detection project. It is intentionally simple and easy to adapt.

## Files

- `train_yolo_template.py` — example training script using Ultralytics YOLO
- `dataset_template.yaml` — dataset configuration template

## Quick start

1. Copy this folder into a new project.
2. Update `dataset_template.yaml` with your image paths and class names.
3. Update the model name, epochs, image size, and batch size in the training script.
4. Run the training script from the project root.

## Example run

```bash
python train_yolo_template.py
```

## Typical project structure

```text
project_name/
  README.md
  dataset/
    images/
    labels/
  dataset.yaml
  scripts/
    train.py
  runs/
```

## Notes

- Keep dataset yaml files versioned with the project.
- Save all important runs under a clear run name.
- Record class definitions and augmentation choices with the experiment.
