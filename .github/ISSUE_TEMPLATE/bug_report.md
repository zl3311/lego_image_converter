---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Load image with `...`
2. Create palette with `...`
3. Run conversion `...`
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```python
from mosaicpic import ConversionSession, Palette, load_image

image = load_image("example.jpg")
palette = Palette.from_set("marilyn_48x48")
session = ConversionSession(image, palette, (48, 48))
session.convert()
# Error occurs here
```

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.12.0]
- mosaicpic version: [e.g., 0.6.0]

## Additional Context
Add any other context, screenshots, or error messages here.
