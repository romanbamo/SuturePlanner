"""
Main Execution Pipeline

Responsibility:
    Handles real-time video capture from a system camera, performs geometric 
    undistortion based on intrinsic camera configurations, and manages asynchronous 
    multithreaded inference processing to visualize surgical incisions and plan suture tracks.

Usage Note:
    Replace placeholders in the "CAMERA CALIBRATION CONFIGURATION" block with your own hardware 
    calibration data before running the module.
"""

import numpy as np
import cv2
import threading
import queue
import time
from YOLO import incision_detector
from Contours import contours
from MidLine import mid_line
from Suture import suture
from CameraMetric import MetricCalculator

# =========================================================================
# CAMERA CALIBRATION CONFIGURATION (PLACEHOLDERS FOR GENERALIZATION)
# =========================================================================
# TODO: Replace with your specific 3x3 Intrinsic Matrix (e.g., from a 1280x720 calibration setup)
K = np.array([[1.0e+03, 0.0e+00, 6.4e+02],
              [0.0e+00, 1.0e+03, 3.6e+02],
              [0.0e+00, 0.0e+00, 1.0e+00]], dtype=np.float32)

# TODO: Replace with your specific Lens Distortion Coefficients vector
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# =========================================================================
# OPERATIONAL CONFIGURATION
# =========================================================================
GRADE = -1
SUTURE_LENGTH = 5

Z_MECHANICAL_VALUE = 450.0  # Distance in mm from sensor/mount tool to plane
Z_OFFSET = 0.0              # Fixed vertical offset adjustments
Z_DEPTH = Z_MECHANICAL_VALUE - Z_OFFSET 

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Precompute optical rectification maps globally 
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (FRAME_WIDTH, FRAME_HEIGHT), alpha=0)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (FRAME_WIDTH, FRAME_HEIGHT), cv2.CV_32FC1)


def inference_worker(input_queue, output_queue, stop_event, calculator, pixel_length):
    """
    PRE-CONDITION:
        - `input_queue` must be a valid queue.Queue container providing rectified image frames.
        - `output_queue` must be a valid queue.Queue container acting as a consumer target.
        - `stop_event` must be a threading.Event instance monitoring system lifecycles.
        - `calculator` must be an initialized MetricCalculator instance.
        - `pixel_length` must be a valid non-zero float value mapping spatial conversion scales.
    POST-CONDITION:
        - Asynchronously polls frames until `stop_event` triggers.
        - Processes available frames through segmentation, contouring, and skeletonization.
        - Places dictionary results containing an annotated overlay array and numerical 
          length values onto `output_queue`.
    """
    while not stop_event.is_set():
        try:
            frame = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        mask = incision_detector.detect_mask(frame, 'mask')
        
        if mask is None:
            result = {
                'overlay': frame,
                'length': 0.0,
            }
        else:
            incision_contours = contours.compute_contours(mask)
            main_contour = contours.get_main_contour(incision_contours)
            line_points = mid_line.thin_skeleton_zhang_suen(mask)
            incision_length = calculator.get_curve_length(line_points)
            suture_data = suture.compute_adaptive_suture(
                line_points, grade=GRADE, margin_length_mm=SUTURE_LENGTH, pixel_size_mm=pixel_length, contour=main_contour
            )
        
            overlay = frame.copy()
            mask_bool = mask > 0.5
        
            if mask_bool.any():
                alpha = 0.4
                overlay[mask_bool] = (
                    frame[mask_bool] * (1 - alpha) +
                    np.array([0, 255, 0]) * alpha
                ).astype(np.uint8)
        
            for pt in line_points:
                cv2.circle(overlay, pt, 1, (255, 0, 0), -1)
        
            for data in suture_data:
                cv2.line(overlay, tuple(data['entry']), tuple(data['exit']), (255, 255, 0), 1)
                cv2.circle(overlay, tuple(data['entry']), 3, (0, 0, 255), -1)
                cv2.circle(overlay, tuple(data['exit']), 3, (255, 0, 0), -1)
        
            result = {
                'overlay': overlay,
                'length': incision_length,
            }
        
        if output_queue.full():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                pass
        output_queue.put(result)


def main():
    """
    PRE-CONDITION:
        - System video input channel indexes must map successfully to a working V4L2 device.
        - Calibration configurations (`K`, `dist`) must be declared or loaded globally.
    POST-CONDITION:
        - Spins up a background worker thread executing pipeline logic.
        - Orchestrates real-time frame capturing, un-distortion mapping, and data queuing.
        - Spawns an interactive high-level OpenCV desktop UI showing data metrics.
        - Releases resource arrays, joins worker threads, and shuts down camera hooks smoothly on terminal 'q' exit keys.
    """
    # TODO: Adjust index (e.g., 4) depending on your target V4L2 device index connection
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    calculator = MetricCalculator(K = K, z_depth = Z_DEPTH)
    pixel_length = calculator.get_pixel_size('x')
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    worker = threading.Thread(
        target=inference_worker,
        args=(input_queue, output_queue, stop_event, calculator, pixel_length),
        daemon=True
    )
    worker.start()
    
    current_overlay = None
    current_length = 0.0
    
    print("Starting video capture... Press 'q' to safely exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image frame from device stream.")
            break
            
        frame_rectificado = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        
        x_r, y_r, w_r, h_r = roi
        if w_r > 0 and h_r > 0:
            frame_final = frame_rectificado[y_r:y_r+h_r, x_r:x_r+w_r]
        else:
            frame_final = frame_rectificado
        
        if input_queue.full():
            try:
                input_queue.get_nowait()
            except queue.Empty:
                pass
        
        input_queue.put(frame_final.copy())
        
        try:
            result = output_queue.get_nowait()
            current_overlay = result['overlay']
            current_length = np.round(result['length'], 2)
        except queue.Empty:
            pass
        
        display_frame = current_overlay if current_overlay is not None else frame_final
        
        cv2.putText(
            display_frame, f"Length: {current_length} mm",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        cv2.imshow('Detected Incision', display_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    stop_event.set()
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    print("Execution finalized successfully.")


if __name__ == "__main__":
    main()