"""
SAM2 Video Segmentation Tracking Pipeline

Responsibility:
    Initializes an Ultralytics SAM2 Video Predictor to track a specific region of interest 
    across video frames using point prompts. Iterates through the tracking sequence to extract 
    and save binary uint8 segmentation masks sequentially onto disk.

Usage Note:
    Update absolute path constants (`MODEL_PATH`, `VIDEO_PATH`, `PROJECT_DIR`, and `OUTPUT_DIR`)
    with your target environment path setup before executing this module on a clean system.
"""

import os
import cv2
import numpy as np
from ultralytics.models.sam import SAM2VideoPredictor

# =========================================================================
# PATH CONFIGURATION PLACEHOLDERS (GENERALIZED FOR REPOSITORY DEPLOYMENT)
# =========================================================================
MODEL_PATH = "path/to/sam2.1_b.pt"
VIDEO_PATH = "path/to/input_video.mp4"
PROJECT_DIR = "path/to/results_project_directory"
OUTPUT_DIR = "path/to/output_masks_directory"

# TODO: Add the coordinates referent to interest areas to segment. Example: [[760, 540]]
INTEREST_POINTS = []
# TODO: Add the label for each coordinate (1 for foreground, 0 for background). Example: [1]
POINT_LABELS = []

os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

overrides = dict(
    conf=0.25, 
    task="segment", 
    mode="predict", 
    imgsz=1024, 
    model=MODEL_PATH, 
    save=True,
    project=PROJECT_DIR,
    name="SAM2_Ultralytics_Video"
)
predictor = SAM2VideoPredictor(overrides=overrides)


def run_video_segmentation_pipeline():
    """
    PRE-CONDITION:
        - `VIDEO_PATH` must point to a valid, accessible video file.
        - `MODEL_PATH` weights file must exist and match SAM2 expectations.
        - `OUTPUT_DIR` must be created and writable.
        - `INTEREST_POINTS` and `POINT_LABELS` must be populated with matching prompt elements.
        - Global `predictor` instantiation must be fully configured.
    POST-CONDITION:
        - Executes SAM2 inference across the target video using user-defined prompt points.
        - Generates tracking result generators sequentially.
        - Isolates structural object layers to write independent binary `uint8` image frames 
          (0 or 255) to `OUTPUT_DIR` with sequential integer index filenames.
    """
    # Fixed the variable name typo (INTEREST_POINTS)
    results = predictor(source=VIDEO_PATH, points=INTEREST_POINTS, labels=POINT_LABELS)
    print("Saving independent binary masks...")
    for frame_idx, result in enumerate(results):
        if result.masks is not None:
            mask_np = result.masks.data[0].cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            
            mask_filename = os.path.join(OUTPUT_DIR, f"{base_name}_frame_{frame_idx:07d}.png")
            cv2.imwrite(mask_filename, mask_uint8)

    print(f"Done! All .png mask files are located in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_video_segmentation_pipeline()