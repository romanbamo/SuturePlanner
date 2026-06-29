# YOLO Training & Specialized Validation Pipeline

## Responsibility
This module governs the optimization, lifecycle training, and cross-dataset validation scripts for structural surgical edge segmentation. It implements a robust, two-phase transfer learning framework that leverages generic features before specializing layers on out-of-distribution targets.

## Directory Structure
```text
 YOLO_train/
├── general_wounds.yaml  # Phase 1: General tissue damage structure mappings
├── incisions.yaml       # Phase 2: Specialized medical incision maps + Test split
├── train.py            # Automated sequential multi-phase execution pipeline
└── YOLO_model/         # Workspace target for base and optimized weight configurations
```
##Training Methodology
The architecture implements a cascading training process designed to maximize generalization across changing clinical scenarios:
- **Phase 1: Pre-training (Transfer Learning)**: Trains a base network (yolo26s-seg) across generic wound topologies (general_wounds.yaml) over 100 epochs using dynamic spatial augmentations to establish steady weight baselines.
- **Phase 2: Fine-Tuning (Domain Adaptation)**: Takes optimal weights from Phase 1 and freezes early layers while running 150 highly augmented epochs (incorporating advanced regularization like Mixup, Copy-Paste, and HSV color space shifts) explicitly focused on clean incision boundaries (incisions.yaml).
- **Phase 3: Out-of-Distribution Independent Audit**: Automatically runs cross-dataset evaluation against a completely separate test subset (e.g., porcine tissue configurations) that was strictly excluded from all training cycles, outputting real, un-biased $mAP_{50}$ and $mAP_{50-95}$ validation metrics.

##Workspace Adjustments
To train the model on a clean system, structure your raw dataset paths and assign them inside `train.py`:
```python
DATASET_BASE_DIR = "path/to/Dataset"
YAML_PHASE1_PATH = "path/to/general_wounds.yaml"
YAML_PHASE2_PATH = "path/to/incisions.yaml"
BASE_MODEL_WEIGHTS = "path/to/yolo26s-seg.pt"
```