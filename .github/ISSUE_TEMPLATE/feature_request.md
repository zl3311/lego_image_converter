---
name: Feature Request
about: Suggest a new feature or enhancement
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Problem Statement
A clear description of the problem or limitation you're facing.

## Proposed Solution
Describe the feature you'd like to see.

## Use Case
Explain how this feature would help your workflow.

## Example Code (Optional)
```python
# How you envision the API working
from legopic import ConversionSession, Palette, load_image

image = load_image("my_photo.jpg")
palette = Palette.from_set(31197)
session = ConversionSession(image, palette, (48, 48))

# Your proposed feature
session.new_feature()  # <- Describe what this would do
```

## Alternatives Considered
Any alternative solutions or workarounds you've tried.

## Additional Context
Any other context, mockups, or references.

