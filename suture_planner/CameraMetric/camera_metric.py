"""
Metric Calculation Module

Responsibility:
    Provides a coordinate conversion and measurement engine based on camera intrinsic 
    parameters and working distance (Z). Calculates spatial pixel lengths, area dimensions 
    in mm² for binary masks, and physical arc-lengths along ordered trajectories.
"""

import numpy as np
import cv2

class MetricCalculator:
    def __init__(self, K, z_depth):
        """
        PRE-CONDITION:
            - `K` must be a valid 3x3 camera intrinsic matrix or array-like structure.
            - `z_depth` must be a numeric value representing the focal distance to the plane.
        POST-CONDITION:
            - Initializes instance variables `fx`, `fy`, and `z_depth`.
            - Precomputes the scaling property `pixel_area_mm2` mapping pixels to square millimeters.
        """
        self.K = np.array(K, dtype=np.float32)
        self.z_depth = float(z_depth)
        
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        
        self.pixel_area_mm2 = (self.z_depth / self.fx) * (self.z_depth / self.fy)

    def get_pixel_size(self, axis=None):
        """
        PRE-CONDITION:
            - `axis` must be either a string ('x', 'X', 'y', 'Y') or None.
        POST-CONDITION:
            - Returns a float if `axis` matches 'x' or 'y' representing the specific axis spatial scale.
            - Returns a tuple `(size_x, size_y)` if `axis` is None.
        """
        size_x = self.z_depth / self.fx
        size_y = self.z_depth / self.fy
        
        if axis == 'x' or axis == 'X':
            return size_x
        elif axis == 'y' or axis == 'Y':
            return size_y
        else:
            return size_x, size_y

    def get_mask_area(self, mask):
        """
        PRE-CONDITION:
            - `mask` must be a valid 2D numpy.ndarray or None.
        POST-CONDITION:
            - Returns a float representing the physical surface area of active pixels in mm².
            - Returns 0.0 if `mask` is None or contains no non-zero elements.
        """
        if mask is None:
            return 0.0
            
        area_pixeles = cv2.countNonZero(mask)
        if area_pixeles == 0:
            return 0.0
            
        return area_pixeles * self.pixel_area_mm2

    def get_curve_length(self, ordered_points):
        """
        PRE-CONDITION:
            - `ordered_points` must be a sequence of 2D coordinates `(x, y)` or None.
        POST-CONDITION:
            - Returns a float tracking the cumulative metric distance in millimeters using Euclidean 
              distance accumulation across scaled dimensional deltas.
            - Returns 0.0 if the sequence is empty or contains fewer than 2 coordinates.
        """
        if ordered_points is None or len(ordered_points) < 2:
            return 0.0

        longitud_total_mm = 0.0

        for i in range(len(ordered_points) - 1):
            pt_actual = ordered_points[i]
            pt_siguiente = ordered_points[i + 1]

            dx_px = pt_siguiente[0] - pt_actual[0]
            dy_px = pt_siguiente[1] - pt_actual[1]

            dx_mm = (dx_px * self.z_depth) / self.fx
            dy_mm = (dy_px * self.z_depth) / self.fy

            longitud_total_mm += np.sqrt(dx_mm**2 + dy_mm**2)

        return longitud_total_mm
        

    def update_z_depth(self, new_z):
        """
        PRE-CONDITION:
            - `new_z` must be a numeric value representing a new distance measurement.
        POST-CONDITION:
            - Overwrites the existing `z_depth` property and updates dependent metrics (`pixel_area_mm2`).
        """
        self.z_depth = float(new_z)
        self.pixel_area_mm2 = (self.z_depth / self.fx) * (self.z_depth / self.fy)