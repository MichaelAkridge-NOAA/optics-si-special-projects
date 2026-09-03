# Templates

This directory holds reusable starting points for new projects and new team members.

## Available templates

- `yolo_training_template/` — a starter YOLO training workflow for a new detection task

## How to use a template

1. Copy the template folder into a new project area.
2. Rename the files to match the project.
3. Update the dataset YAML, class names, and project metadata.
4. Replace placeholder model names and run settings.
5. Save outputs under a project-specific `runs/` or `training_runs/` folder.

## Recommended naming pattern

```text
<project>/<version>/
  dataset/
  training_runs/
  notebooks/
  scripts/
  README.md
```

This keeps each team project reproducible and easier to review.
