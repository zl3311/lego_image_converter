# Releasing

This project uses [Semantic Versioning](https://semver.org/) and automated PyPI publishing via GitHub Actions.

## Quick Reference

```bash
# 1. Update version
# Edit pyproject.toml and src/mosaicpic/__init__.py

# 2. Commit version bump
git add pyproject.toml src/mosaicpic/__init__.py
git commit -m "chore: Bump version to X.Y.Z"
git push origin main

# 3. Wait for CI to pass ✓

# 4. Create and push tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Version Types

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fixes | Patch | `0.6.0` → `0.6.1` |
| New features (backward compatible) | Minor | `0.6.0` → `0.7.0` |
| Breaking changes | Major | `0.6.0` → `1.0.0` |

## What Happens Automatically

When you push a tag matching `v*.*.*`:

1. **Build** — Package is built with `python -m build`
2. **Publish** — Uploaded to [PyPI](https://pypi.org/project/mosaicpic/)
3. **GitHub Release** — Created with auto-generated release notes

## Files to Update

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/mosaicpic/__init__.py` | `__version__ = "X.Y.Z"` |

## Verify Before Release

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/mosaicpic/
uv run pytest
```
