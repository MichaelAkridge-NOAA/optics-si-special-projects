# Project Team Workspaces(folders)

This folder is intended for shared team-level materials that apply across projects.

## Purpose

Keep team-wide context in one place so that each project can stay focused on its own data, training runs, and experiments while still following the same collaboration conventions.

## Suggested contents

- team norms and operating rhythm
- review checklist and milestone notes
- shared model evaluation standards
- cross-project documentation and links
- archive notes for recurring workflows

## Recommended structure

```text
projects/
  README.md
  standards/
    SOP.md
    model_training_checklist.md
    dataset_review_checklist.md
  meetings/
    sprint_notes.md
  archive/
    historical_notes.md
    old_script_example.py
```

## Best practices

- Keep shared docs concise and reusable.
- Use project folders for data and experiments.
- Preserve training metadata, dataset splits, and run notes with each experiment.
- Avoid duplicating project-specific logic in this folder.

## Team default workflow

1. Start with a project folder or a template project.
2. Ensure the dataset review is documented.
3. Save train/validation splits and labels alongside the data.
4. Run experiments with explicit naming and artifact folders.
5. Record the final configuration and evaluation summary.
