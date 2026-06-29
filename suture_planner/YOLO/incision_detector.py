"""
Incision Detection Module

Responsibility:
    Loads a custom YOLO segmentation model into memory and executes inference on video frames 
    to extract binary masks or visual overlays of surgical incisions.
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'last_v6.pt')
model = YOLO(model_path)


def detect_mask(img, ret='mask'):
    """
    PRE-CONDITION:
        - `img` must be a valid, non-empty numpy.ndarray representing a BGR image frame.
        - `ret` must be a string, either 'mask' or 'plot'.
        - The global `model` object must be initialized with its neural weights.
    POST-CONDITION:
        - If `ret` == 'plot', returns a numpy.ndarray containing the frame with visual segmentation overlays.
        - If `ret` == 'mask' and an incision is found, returns a binary 2D numpy.ndarray (uint8) of the 
          same width and height as `img`, where the incision is drawn in white (255) over a black background.
        - If no incision is detected, returns None.
        - The internal execution logic, inference confidence thresholds, and spatial dimensions remain unaltered.
    """
    original_h, original_w = img.shape[:2]
    
    results = model.predict(source=img, device=device, conf=0.617, imgsz=640, verbose=False)
    r = results[0]

    if ret == 'plot':
        return r.plot(labels=False, boxes=False, conf=False)
    
    if ret == 'mask':
        if r.masks is not None:
            mask_out = np.zeros((original_h, original_w), dtype=np.uint8)
        
            if r.masks.xy and len(r.masks.xy[0]) > 0:
                contour = r.masks.xy[0].astype(np.int32)
                cv2.fillPoly(mask_out, [contour], 255)
        
            return mask_out
        else:
            return None

    return None
