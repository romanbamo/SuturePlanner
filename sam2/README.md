# SAM2 Video Mask Generator

## Responsibility
This module handles autonomous object tracking and dataset generation across surgical video sequences. Utilizing the **Segment Anything Model 2 (SAM 2)** architecture from Ultralytics, it tracks target regions of interest over time based on zero-shot interactive point prompts, isolating and extracting continuous semantic structures into binary image representations.

## Directory Structure
```text
|- sam2/
└── SAM_mask_generator.py   # Main predictor pipeline utilizing tracking generators
```

## Features

- **Zero-Shot Spatio-Temporal Tracking**: Propagates segmented bounds frame-by-frame across high-resolution surgical streams.

- **Coordinate Prompt Integration**: Accepts user-defined multi-point pixel coordinates (`INTEREST_POINTS`) and corresponding tracking foreground/background indicators (`POINT_LABELS`).

- **Correlative Frame Extraction**: Sequentially extracts tracking masks and exports them to disk as individual 8-bit single-channel (`uint8`, values 0 or 255) `.png` frames formatted for immediate semantic alignment.

##Core Setup

Before launching execution, open `SAM_mask_generator.py` and modify the environment paths and tracking prompts:

```python
MODEL_PATH = "path/to/sam2.1_b.pt"
VIDEO_PATH = "path/to/input_video.mp4"
PROJECT_DIR = "path/to/results_project_directory"
OUTPUT_DIR = "path/to/output_masks_directory"

# Prompt Configuration Example (Modify according to workspace requirements)
INTEREST_POINTS = [[760, 540]]
POINT_LABELS = [1]
```
