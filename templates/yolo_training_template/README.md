# YOLO Training Template

This template is intended as a starting point for a new detection project. It is intentionally simple and easy to adapt.

## Files

- `train_yolo_template.py` — example training script using Ultralytics YOLO
- `dataset_template.yaml` — dataset configuration template
- `01_cloud_yolo_dataset_prep.ipynb` — acquire images and labels, validate YOLO data, create reproducible splits, and publish the preparation handoff
- `02_cloud_yolo_training.ipynb` — verify the prepared handoff, train with simple or advanced settings, evaluate, resume, and upload artifacts
- `install_cloud_workstation.sh` — repeatable conda, CUDA PyTorch, JupyterLab kernel, and shared-memory setup for a Linux Cloud Workstation

## Quick start

1. Copy this folder into a new project.
2. Update `dataset_template.yaml` with your image paths and class names.
3. Update the model name, epochs, image size, and batch size in the training script.
4. Run the training script from the project root.

For cloud-hosted datasets, use the numbered notebooks in order:

1. Run `01_cloud_yolo_dataset_prep.ipynb` when images, labels, classes, or split policy change. It supports a fresh Label Studio API export, an existing YOLO export mirrored to GCS, or a YOLO ZIP uploaded through JupyterLab.
2. Review the validation report and labeled-image previews. The notebook writes `dataset.yaml`, `split_manifest.json`, and `prep_summary.json` under `workspace/<project>`.
3. Open `02_cloud_yolo_training.ipynb`, configure the same project/workspace, and verify the preparation handoff before training.
4. Start with simple training mode. Enable advanced settings only after establishing a baseline.

The training notebook never downloads labels or rebuilds a split. If data changes, return to notebook 1 and create a new preparation handoff.

## Google Cloud Workstation setup

Run these commands in a JupyterLab terminal on the Linux workstation. Download the script first so it can be reviewed before execution:

```bash
cd ~
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
wget --no-cache -O install_cloud_workstation.sh \
  https://raw.githubusercontent.com/MichaelAkridge-NOAA/optics-si-special-projects/main/templates/yolo_training_template/install_cloud_workstation.sh
chmod +x install_cloud_workstation.sh
./install_cloud_workstation.sh
```

The default setup clones or fast-forward pulls this repository into `~/optics-si-special-projects`, creates a Python 3.12 conda environment named `yolo-cloud`, installs CUDA 12.4 PyTorch plus the notebook dependencies, registers a Jupyter kernel, and requests a 16 GB `/dev/shm` mount. It is safe to rerun and reuses the conda environment.

Customize it with environment variables when needed:

```bash
# Different environment name, Python version, and shared-memory size
ENV_NAME=serdp-yolo PYTHON_VERSION=3.11 SHM_SIZE=24G \
  ./install_cloud_workstation.sh

# Skip the shared-memory remount when sudo or mount privileges are unavailable
UPDATE_SHM=false ./install_cloud_workstation.sh

# Skip apt-based system package setup when sudo is unavailable
INSTALL_SYSTEM_PACKAGES=false ./install_cloud_workstation.sh

# Use a different NumPy constraint if a project needs it
NUMPY_SPEC='numpy>=2.2,<2.3' ./install_cloud_workstation.sh

# Use CPU-only PyTorch on a workstation without an NVIDIA GPU
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  ./install_cloud_workstation.sh

# Clone or update the repository somewhere else
REPO_DIR=~/projects/optics-si-special-projects ./install_cloud_workstation.sh

# Explicitly accept Anaconda Terms of Service during setup
ACCEPT_ANACONDA_TOS=true ./install_cloud_workstation.sh
```

The installer uses `apt-get` when available to install common workstation tools and native libraries (`git`, `wget`, `unzip`, `zip`, `libgl1`, `libglib2.0-0`, `libsm6`, and `libxext6`). It then uses `conda create`, upgrades `pip`, installs PyTorch separately from the selected PyTorch wheel index, and installs Ultralytics, PyYAML, Requests, Pillow, Matplotlib, NumPy, pandas, Label Studio SDK, Google Cloud Storage, ipykernel, and JupyterLab. It avoids cached pip wheels, force-reinstalls NumPy with `NUMPY_SPEC='numpy>=2.2,<2.3'` by default, runs `pip check`, and verifies the notebook imports before reporting setup complete.

If verification fails with a NumPy compiled-extension error such as `cannot read file data`, rerun the installer. It will reuse the existing conda environment and repair NumPy with a fresh non-cached wheel. To run just the repair step manually:

```bash
conda run -n yolo-cloud python -m pip install --force-reinstall --no-cache-dir 'numpy>=2.2,<2.3'
conda run -n yolo-cloud python -c 'import numpy; print(numpy.__version__)'
```

If `conda create` stops with `CondaToSNonInteractiveError`, review and accept the Anaconda channel Terms of Service, then rerun the installer:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
./install_cloud_workstation.sh
```

After setup:

1. Refresh JupyterLab.
2. Open `01_cloud_yolo_dataset_prep.ipynb`.
3. Select **Kernel > Change Kernel > Python (yolo-cloud)**, or the display name matching `ENV_NAME`.
4. Use the same kernel for `02_cloud_yolo_training.ipynb` and confirm it reports `CUDA available: True` when using a GPU workstation.

If a notebook reports a missing package such as `ModuleNotFoundError: No module named 'matplotlib'`, it is usually using the wrong Python kernel. Run this in a notebook cell:

```python
import sys
print(sys.executable)
```

The path should include the conda environment name, for example `yolo-cloud`. If it does not, select **Kernel > Change Kernel > Python (yolo-cloud)**, then restart the notebook kernel and rerun the cells. If the kernel is not listed, run:

```bash
conda run -n yolo-cloud python -m ipykernel install --user \
  --name yolo-cloud \
  --display-name "Python (yolo-cloud)"
jupyter kernelspec list
```

### Helpful workstation commands

Use these in a JupyterLab terminal on the Linux workstation:

```bash
# Go to the cloned project
cd ~/optics-si-special-projects

# Pull the latest template and project updates
git pull --ff-only

# Make folders for uploads, outputs, or a new project workspace
mkdir -p uploads workspace/my_project

# Copy this template into a new project folder
cp -r templates/yolo_training_template projects/my_new_project

# List files, including sizes and hidden files
ls -lah

# Unzip a Label Studio or YOLO export into a folder
unzip uploads/export.zip -d uploads/export

# Zip a run folder for download or archive
zip -r yolo_run_archive.zip workspace/my_project/runs

# Check available disk space and shared memory
df -h
df -h /dev/shm

# Activate the environment in a terminal shell
conda activate yolo-cloud

# Start JupyterLab if it is not already running
jupyter lab --ip=0.0.0.0 --no-browser
```

### Shared memory notes

Check the current shared-memory allocation with:

```bash
df -h /dev/shm

sudo mount -o remount,size=8G /dev/shm
```

The script runs `mount -o remount,size=... /dev/shm` through root or `sudo`. Managed Cloud Workstation containers may not grant mount privileges. The remount is also temporary on many images and is lost after a workstation or container restart. For a persistent setting, update the workstation container/runtime configuration or ask the platform administrator to configure the shared-memory mount. Lower the notebook's `workers` value if shared memory remains constrained.

### Google Cloud authentication

Authenticate in the JupyterLab terminal rather than a notebook cell:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth list
```

For unattended jobs, prefer the workstation's attached service account or workload identity. Do not store service-account keys, API tokens, or other credentials in the repository or notebook.

### Label sources

Set `LABEL_SOURCE_MODE` in the notebook configuration cell:

- `label_studio_api` downloads a fresh YOLO export using `LABEL_STUDIO_API_TOKEN` from the environment.
- `gcs_export` downloads a YOLO ZIP or directory from `LABEL_EXPORT_GCS_URI`.
- `user_upload_zip` reads a local ZIP uploaded through JupyterLab. Create an `uploads` folder in the file browser, upload the export, and set `USER_LABEL_ZIP` to its local path.

All ZIP sources use protected extraction and the same image/label integrity checks before training.

## Example run

```bash
python train_yolo_template.py
```

## Typical project structure

```text
project_name/
  README.md
  01_cloud_yolo_dataset_prep.ipynb
  02_cloud_yolo_training.ipynb
  workspace/
    project_name/
      dataset.yaml
      split_manifest.json
      prep_summary.json
      dataset/
        images/
        labels/
      runs/
```

## Notes

- Keep dataset yaml files versioned with the project.
- Save all important runs under a clear run name.
- Record class definitions and augmentation choices with the experiment.
