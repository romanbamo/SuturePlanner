"""
Contours Processing Module

Responsibility:
    Extracts geometric boundaries from binary incision masks, identifies the primary 
    structural contour by area, and filters out isolated background noise or artifacts.
"""

import cv2
import numpy as np

def compute_contours(mask):
    """
    PRE-CONDITION:
        - `mask` must be a valid 2D numpy.ndarray (uint8) representing a binary mask.
    POST-CONDITION:
        - Returns a tuple or list of numpy.ndarrays representing the extracted extreme 
          outer contours.
        - The original input `mask` is left completely unmodified.
    """
    mask_copy = mask.copy()
    
    contours, _ = cv2.findContours(
        mask_copy, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours
    

def get_main_contour(contours):
    """
    PRE-CONDITION:
        - `contours` must be a non-empty sequence containing valid contour structures 
          extracted via OpenCV.
    POST-CONDITION:
        - Returns a single numpy.ndarray representing the contour with the largest 
          calculated spatial area using `cv2.contourArea`.
    """
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def draw_main_contour(mask, contours):
    """
    PRE-CONDITION:
        - `mask` must be a valid 2D numpy.ndarray (uint8) serving as the baseline shape reference.
        - `contours` must be a sequence of contour arrays (can be empty).
    POST-CONDITION:
        - If `contours` is empty, returns the original unmodified `mask`.
        - If `contours` contains elements, returns a new black binary mask (zeros_like) 
          of the exact same dimensions, with only the interior of the single largest contour 
          filled with white pixels (255).
    """
    if not contours:
        return mask
    
    filled_mask = np.zeros_like(mask)
    largest_contour = max(contours, key=cv2.contourArea)
    
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=-1) 

    return filled_mask
