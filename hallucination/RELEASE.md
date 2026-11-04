# Release Process

This repository ships a single distribution, `hallucination-toolkit`, containing
both subpackages (`hallucination.detection`, `hallucination.mitigation`).
Version source: `pyproject.toml` at the repository root. One version covers the
whole toolkit; the CHANGELOG notes which subpackage changed.

Tag scheme: plain `vX.Y.Z` on the repository root. (The historical
`detection-v0.3.0` / `mitigation-v0.1.0` tags refer to the pre-merge
two-package layout and are kept for history.)

## Pre-release checklist

Run from the repository root (CI runs the same steps on every push):

```bash
python -m ruff check src tests web_demo
python -m pytest tests
```

## Cutting a release

1. Update `version` in `pyproject.toml` and `src/hallucination/_version.py`
   (keep the two in sync).
2. Move the CHANGELOG section from *Unreleased* to a dated section. The date
   is the day the tag is cut, even if the PyPI upload happens later.
3. Build and validate locally:

   ```bash
   python -m pip install -U build twine
   python -m build            # wheel + sdist
   python -m twine check dist/*
   ```

4. Commit the version bump and changelog, then tag:

   ```bash
   git tag -a v0.1.0 -m "hallucination-toolkit 0.1.0"
   git push origin main v0.1.0
   ```

5. Upload to PyPI from a clean tree:

   ```bash
   python -m twine upload dist/*
   ```

   PyPI requires a verified e-mail address on the uploading account and an
   API token (`--username __token__`).

## Notes

- Never publish from a dirty tree, and never publish artifacts built from a
  commit that is not tagged.
- Before the first public upload, confirm the `hallucination-toolkit` name is
  available (or reserved) on PyPI.
