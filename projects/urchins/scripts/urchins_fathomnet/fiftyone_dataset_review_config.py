# view_urchins.py
# pip install fiftyone pillow

import json
from pathlib import Path
import fiftyone as fo
import fiftyone.core.labels as fol
from fiftyone import ViewField as F

# --- config you may tweak ---
PORT = 5151  # tunnel this to your laptop
ROOT = Path(r"U:\OTHER\AI_DATASETS\fathomnet\20251002_urchins")
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
MAP_FILE   = ROOT / "classes.json"
DS_NAME = "fathomnet_urchins"

# --- sanity checks ---
assert IMAGES_DIR.exists(), f"Missing images dir: {IMAGES_DIR}"
assert LABELS_DIR.exists(), f"Missing labels dir: {LABELS_DIR}"
assert MAP_FILE.exists(),   f"Missing class map: {MAP_FILE}"

# Make FiftyOne prefer a fixed port + local bind (secure; SSH tunnel it)
fo.config.default_app_port = PORT
fo.config.launch_app_locally = True  # binds 127.0.0.1

# --- classes map ---
with open(MAP_FILE, "r") as f:
    concept_to_id = json.load(f)
id_to_concept = {int(v): k for k, v in concept_to_id.items()}

# --- dataset (re)create ---
if DS_NAME in fo.list_datasets():
    fo.delete_dataset(DS_NAME)
ds = fo.Dataset(DS_NAME, persistent=True)
ds.default_classes = [id_to_concept[i] for i in sorted(id_to_concept)]
ds.info["concept_to_id"] = concept_to_id

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def parse_yolo_labels(txt_path: Path) -> fol.Detections:
    dets = []
    if not txt_path.exists():
        return fol.Detections(detections=dets)
    with open(txt_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                continue
            x = _clamp01(xc - w / 2.0)
            y = _clamp01(yc - h / 2.0)
            w = _clamp01(w)
            h = _clamp01(h)
            if w <= 0 or h <= 0:
                continue
            label = id_to_concept.get(cls_id, f"class_{cls_id}")
            det = fol.Detection(label=label, bounding_box=[x, y, w, h])
            det["class_id"] = int(cls_id)
            det.tags = ["urchin"]
            dets.append(det)
    return fol.Detections(detections=dets)

# ingest
img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in img_exts)

samples = []
for p in paths:
    uuid = p.stem
    lbl = LABELS_DIR / f"{uuid}.txt"
    sample = fo.Sample(filepath=str(p))
    sample["detections"] = parse_yolo_labels(lbl)
    sample["uuid"] = uuid
    samples.append(sample)

ds.add_samples(samples)
ds.compute_metadata()

empty_view = ds.match(F("detections.detections").length() == 0)
if len(empty_view) > 0:
    empty_view.tag_samples("no_labels")

print(ds)
print("classes:", ds.default_classes)
print("empty (no_labels) samples:", len(empty_view))

# Launch app and block until you close it (Ctrl+C here to stop)
session = fo.launch_app(ds, port=PORT)
session.wait()
