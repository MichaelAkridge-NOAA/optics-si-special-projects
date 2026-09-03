from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_ROOT / "dataset_template.yaml"
MODEL_NAME = "yolov8n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16


def main() -> None:
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_ROOT / "runs"),
        name="train_run",
        exist_ok=True,
        pretrained=True,
    )

    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
