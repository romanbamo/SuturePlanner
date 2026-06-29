# Suture Planner
## Depth Approximation & Camera Calibration Methodology

### Pseudo-3D Spatial Mapping via Manual Z-Axis Control
In the absence of a dedicated three-dimensional or RGB-D depth camera, the system implements a software-driven pseudo-3D spatial approximation engine. 

Instead of deploying dynamic stereoscopic sensors, the pipeline calculates physical millimeter measurements across the working plane by combining a standard monocular webcam input with an explicit, manually-controlled vertical depth coordinate ($Z$).

## Responsibility
This is the core operational package of the real-time pipeline. It coordinates high-speed video streams, performs camera matrix-driven lens undistortion, executes instance/semantic boundary segmentation via customized YOLO models, and applies advanced morphology to perform **adaptive recursive arc-length trajectory planning** for robotic suture placements.

## Architecture & Submodules
```text
 suture_planner/
├── CameraMetric/    # Spatial coordinate mapping and physical metric estimation
├── Contours/        # Tissue boundary tracing and region isolation
├── MidLine/         # Centroid ridge skeletonization (Zhang-Suen)
├── Suture/          # Recursive perpendicular path-planning calculation
├── YOLO/            # Asynchronous inference wrappers for incision detection
└── main.py          # Multithreaded execution and visualization orchestration
```
## Multithreaded Pipeline Execution

To achieve low-latency real-time video streaming without frame drops, the engine detaches heavy deep learning steps into an asynchronous architecture:

- **Main Thread** (UI & I/O Loop): Interfaces with the camera hook, captures input matrices, maps pixel coordinates through precomputed spatial lookup arrays (cv2.initUndistortRectifyMap), populates the input_queue, polls results from output_queue, and draws standard overlays.

- **Inference Worker Thread**: Monitors incoming frame notifications. It concurrently drives YOLO boundary segmentations, applies the Zhang-Suen thinning algorithm, resolves physical dimensions via CameraMetric, maps normal trajectories, and passes completed dictionaries back into the visualization stream.

##Setup & Deployment

Ensure your local hardware workspace dimensions, extrinsic/intrinsic parameters, and system hardware index tags match your system constraints inside main.py:

```python
# 3x3 Intrinsic Calibration Matrix (1280x720)
K = np.array([[1.0e+03, 0.0e+00, 6.4e+02],
              [0.0e+00, 1.0e+03, 3.6e+02],
              [0.0e+00, 0.0e+00, 1.0e+00]], dtype=np.float32)

# Lens Distortion Matrix
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Mechanical Working Depth Constraints
Z_MECHANICAL_VALUE = 450.0  # Vertical distance in mm
Z_OFFSET = 0.0              # Sensor tool alignment adjustments
```