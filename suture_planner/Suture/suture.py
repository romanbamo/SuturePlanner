"""
Suture Path Planning Module

Responsibility:
    Computes optimal, safe robotic entry and exit trajectories perpendicular to the 
    incision centerline. Integrates adaptive physical length triggers (6mm - 8mm) via 
    recursive arc-length subdivision and performs edge-detection ray-marching to offset 
    stitch locations externally relative to the outer tissue contours.
"""

import numpy as np
import cv2

def _get_local_orthogonal_vector(line_segment, windows, mid_idx=None):
    """
    PRE-CONDITION:
        - `line_segment` must be a sequence of 2D coordinates `(x, y)` tracking the centerline path.
        - `windows` must be an integer specifying the indexing step range for tangent calculation.
        - `mid_idx` is either an integer index or None.
    POST-CONDITION:
        - Returns a 1D numpy.ndarray (float64) of size 2 containing a normalized orthogonal unit vector.
        - Returns `[0, 0]` if the vector magnitude equals zero to prevent division by zero errors.
    """
    if mid_idx is None:
        mid_idx = len(line_segment) // 2
        
    window = windows
    start_idx = max(0, mid_idx - window)
    end_idx = min(len(line_segment) - 1, mid_idx + window)
    
    p1 = np.array(line_segment[start_idx])
    p2 = np.array(line_segment[end_idx])
    
    direction_vector = p2 - p1
    orthogonal_vector = np.array([-direction_vector[1], direction_vector[0]], dtype=np.float64)
    
    norm = np.linalg.norm(orthogonal_vector)
    if norm > 0:
        return orthogonal_vector / norm
    
    return np.array([0, 0])


def _find_contour_intersection(start_point, unit_vector, contour):
    """
    PRE-CONDITION:
        - `start_point` must be a coordinate array/list `[x, y]` representing the ray origin.
        - `unit_vector` must be a pre-normalized 2D directional array of magnitude 1.
        - `contour` must be a valid numpy.ndarray containing polygon coordinates.
    POST-CONDITION:
        - Returns a 1D numpy.ndarray (float64) with the exact sub-pixel coordinate `[x, y]` 
          where the advanced vector exits the outer boundaries of the contour (`dist < 0`).
        - Safety-bounded to automatically return coordinates if ray propagation crosses 
          outside a hardcoded [0, 2000] pixel grid boundary limit.
    """
    current_pt = np.array(start_point, dtype=np.float64)
    step_size = 0.5
    
    while True:
        current_pt += unit_vector * step_size
        dist = cv2.pointPolygonTest(contour, (float(current_pt[0]), float(current_pt[1])), False)
        
        if dist < 0:
            return current_pt
        
        if current_pt[0] < 0 or current_pt[0] > 2000 or current_pt[1] < 0 or current_pt[1] > 2000:
            return current_pt


def compute_adaptive_suture(midline, grade, margin_length_mm, pixel_size_mm, contour, windows=10, results=None):
    """
    PRE-CONDITION:
        - `midline` must be a sequence of 2D coordinates representing a continuous ordered centerline path.
        - `grade` must be an integer. A value of -1 enables structural physical millimeter-based 
          adaptive split constraints, whereas fixed positive integers determine classic recursion steps.
        - `margin_length_mm` must be a numeric value defining the desired needle offset in millimeters.
        - `pixel_size_mm` must be a float indicating the spatial scaling factor (mm/pixel).
        - `contour` must be a valid array representing the master target mask shape.
    POST-CONDITION:
        - Returns a structured list of dictionaries, where each entry contains integer lists mapping 
          'center', 'entry', and 'exit' coordinates in pixel dimensions.
        - If the length of `midline` is fewer than 2 elements, returns the accumulated `results` array immediately.
        - Modifies the accumulated `results` list in place during recursive branch execution.
    """
    if results is None:
        results = []

    if len(midline) < 2:
        return results

    pts = np.array(midline)
    diffs = np.diff(pts, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    total_length_pixels = np.sum(distances)
    total_length_mm = total_length_pixels * pixel_size_mm

    if grade == -1:
        if 6.0 <= total_length_mm <= 8.0:
            grade = 1
        elif total_length_mm < 6.0:
            return results
    elif grade <= 0:
        return results

    cumulative_length = np.insert(np.cumsum(distances), 0, 0.0)
    half_length = total_length_pixels / 2.0
    mid_idx = np.argmin(np.abs(cumulative_length - half_length))
    mid_point = pts[mid_idx]
    
    unit_orthogonal = _get_local_orthogonal_vector(midline, windows, mid_idx)
    
    if np.all(unit_orthogonal == 0):
        return results

    intersection_entry = _find_contour_intersection(mid_point, unit_orthogonal, contour)
    intersection_exit = _find_contour_intersection(mid_point, -unit_orthogonal, contour)
    
    margin_length_pixels = margin_length_mm / pixel_size_mm
    
    entry_point = intersection_entry + (unit_orthogonal * margin_length_pixels)
    exit_point = intersection_exit + (-unit_orthogonal * margin_length_pixels)
    
    results.append({
        'center': mid_point.astype(int).tolist(),
        'entry': entry_point.astype(int).tolist(),
        'exit': exit_point.astype(int).tolist()
    })

    if grade > 1 or grade == -1:
        next_grade = -1 if grade == -1 else grade - 1
        
        compute_adaptive_suture(midline[:mid_idx+1], next_grade, margin_length_mm, pixel_size_mm, contour, windows, results)
        compute_adaptive_suture(midline[mid_idx:], next_grade, margin_length_mm, pixel_size_mm, contour, windows, results)

    return results