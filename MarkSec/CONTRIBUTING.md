# Contributing to MarkSec

Thank you for contributing to MarkSec. The project is a configuration-driven
toolkit for evaluating attacks against LLM watermarking systems, so
contributions should keep experiments reproducible and easy to audit.

## Development Setup

Use Python 3.11 or newer:

```bash
conda create -n wm python=3.11 -y
conda activate wm
pip install -r requirements.txt
```

Optional extras are declared in `pyproject.toml`:

```bash
pip install -e .[dev]
pip install -e .[logging]
pip install -e .[db-mysql]
pip install -e .[db-postgres]
```

## Design Principles

- Keep experiments runnable from config files.
- Preserve CLI override compatibility with `-o key=value`.
- Load heavyweight models through the existing manager/factory path.
- Keep attack-specific parameters in `configs/attack/<ATTACK>.json`.
- Write generated artifacts under `output/`, which is ignored by git.

## Adding an Attack

1. Add an attack config under `configs/attack/<ATTACK_KEY>.json`.
2. Implement the attack under `src/attack/<module_name>/`.
3. Export public classes from that module's `__init__.py`.
4. Register the attack in `src/attack/factory.py`.
5. Add or update focused tests where practical.

Attack classes should extend the existing base classes and receive models,
tokenizers, datasets, watermark instances, and devices through factory
injection instead of loading them ad hoc.

## Checks Before Opening a PR

Run the checks that match your change:

```bash
ruff check .
ruff format --check .
```

If you add or modify tests, run the relevant `pytest` target as well. Large
model tests are expensive and environment-dependent. For changes that touch GPU
execution, include the exact config, model cache setup, and command you used for
validation in the pull request.

## Pull Request Notes

Please include:

- What changed and why.
- Which benchmark settings or configs are affected.
- Commands used to validate the change.
- Any new external model, dataset, solver, or API dependency.

Do not commit secrets, machine-local paths, downloaded model weights, generated
outputs or database files.
