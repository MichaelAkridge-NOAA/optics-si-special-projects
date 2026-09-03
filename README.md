# Optics SI Special Projects

<a href="" target="_blank"><img src="./docs/logo/optics_si_logo_v1.png" align="right" alt="Optics SI logo" height="220"/></a>

This repository is organized as a shared team workspace for marine imagery and computer vision projects. It supports multiple active detection efforts while also providing reusable templates and shared team materials for new work.

## Mission

The goal of this workspace is to support reproducible, reviewable, and collaborative model development for underwater imagery tasks such as species detection, bleaching assessment, and habitat monitoring.

## Repository Layout

```text
optics-si-special-projects/
  README.md
  projects/
    README.md
    bleaching/
    ESA/
    SERDP/
    urchins/
    standards/
  templates/
    README.md
    yolo_training_template/
  docs/
  scripts/
```

## Shared Project Structure

### `project/`
This is the shared home for cross-project team materials.

- common standards and review checklists
- team documentation and process notes
- shared collaboration resources
- archive and planning materials

See [teams/README.md](teams/README.md) for the default structure and guidance.

### `templates/`
This folder is intended for reusable project starting points.

- a standard YOLO training starter
- simple project scaffolding for new teams
- repeatable training and organization patterns

See [templates/README.md](templates/README.md) for the default template setup.

---

## Active Projects

### Urchins
A multi-class detector for identifying important urchin taxa in benthic imagery.

- Focus: ecological monitoring and species classification
- Classes: CHGI, DISP, ECMA, ECST, ECTH, EUME, HEMA, PAGR, TRGR
- Workspace: `teams/urchins/multi-class/`
- Active version: `teams/urchins/multi-class/v1/`

### SERDP
A single-class detector for invasive species monitoring within the SERDP effort.

- Focus: early identification and monitoring of invasive taxa
- Current area: `teams/SERDP/unomia/`

### ESA
Environmental species assessment models for coral monitoring and classification tasks.

#### AGLO Model
- Single-class detector
- Target class: AGLO
- Workspace: `teams/ESA/ESA_AGLO/`
#### ICRA Model
- Single-class detector
- Target class: ICRA
- Workspace: `teams/ESA/ESA_ICRA/`
- Links:
  - Model card: https://huggingface.co/NMFS-OSI/yolo11m-esa-coral-icra-detector
  - Training dataset: https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-ESA-CORAL-ICRA-Dataset
  - Model demo: https://huggingface.co/spaces/NMFS-OSI/ESA-Coral-ICRA-Detector-Demo

### Bleaching
A reef condition modeling workflow focused on bleaching-related classes.

- Focus: bleaching-state detection and assessment
- Workspace: `teams/bleaching/three-class-model/`

---

## Recommended Team Workflow

1. Start from a project folder or a template in `templates/`.
2. Prepare and review the dataset before training.
3. Keep splits, labels, and metadata with the project.
4. Train using consistent naming and output folders.
5. Archive results and notes for reproducible review.

## Suggested Starting Points
- ESA work: `project/ESA/`
- New project template: `templates/yolo_training_template/`

---

## Status

This repo is structured as a shared collaborative workspace for special-project model development. The new team and template folders are intended to make onboarding easier and keep experiments consistent across projects.


