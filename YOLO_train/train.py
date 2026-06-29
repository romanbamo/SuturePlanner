"""
YOLO Multi-Phase Training and Validation Pipeline

Responsibility:
    Orchestrates a robust, two-phase training strategy (transfer learning and fine-tuning) 
    for surgical incision segmentation. Automatically builds workspace directory architectures, 
    manages dataset distributions, executes sequential training phases, and evaluates 
    the resulting weights using an independent, out-of-distribution external test split.

Usage Note:
    Update absolute path placeholders inside `pipeline_completo_local()` to map 
    the local project dataset environment before deploying this engine.
"""

import os
import shutil
from ultralytics import YOLO

def prepare_dir_structure(base_dir_placeholder="path/to/Dataset"):
    """
    PRE-CONDITION:
        - `base_dir_placeholder` must be a string representing a valid workspace directory route.
    POST-CONDITION:
        - Creates a structured tree of directories for training, validation, and testing splits 
          if they do not already exist on disk.
    """
    dirs_phase1 = [
        f"{base_dir_placeholder}/General_Wounds_DataSet/images/train",
        f"{base_dir_placeholder}/General_Wounds_DataSet/images/val",
        f"{base_dir_placeholder}/General_Wounds_DataSet/labels/train",
        f"{base_dir_placeholder}/General_Wounds_DataSet/labels/val"
    ]

    dirs_phase2 = [
        f"{base_dir_placeholder}/Incisions_DataSet/images/train",
        f"{base_dir_placeholder}/Incisions_DataSet/images/val",
        f"{base_dir_placeholder}/Incisions_DataSet/labels/train",
        f"{base_dir_placeholder}/Incisions_DataSet/labels/val",
        f"{base_dir_placeholder}/Validation_DataSet/images",
        f"{base_dir_placeholder}/Validation_DataSet/labels"
    ]

    for dir_path in dirs_phase1 + dirs_phase2:
        os.makedirs(dir_path, exist_ok=True)


def phase1_pretrain(model_path, yaml_path):
    """
    PRE-CONDITION:
        - `model_path` must point to a valid, reachable Ultralytics base segmentation model file (.pt).
        - `yaml_path` must point to a structured dataset YAML configuration for general wounds.
    POST-CONDITION:
        - Performs transfer learning training over 100 epochs with geometric augmentations.
        - Returns a string mapping the absolute file path to the generated optimal weights (`best.pt`).
    """
    print("\n" + "="*60)
    print(" PHASE 1: PRE-TRAINING ON GENERAL WOUNDS")
    print("="*60)

    model = YOLO(model_path)
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="TFG_Incisions",
        name="phase1_general_wounds",
        save=True,
        plots=True,
        patience=20,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        degrees=90.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.8,
    )
    return f"{results.save_dir}/weights/best.pt"


def phase2_finetuning(model_path_phase1, yaml_path):
    """
    PRE-CONDITION:
        - `model_path_fphase1` must point to the best weights generated during Phase 1 operations.
        - `yaml_path` must point to a structured dataset YAML mapping specific surgical incisions.
    POST-CONDITION:
        - Executes specialized fine-tuning for 150 epochs using optimized hyper-parameters, 
          regularization dropouts, and extensive augmentation parameters (mosaic, mixup, copy-paste).
        - Returns a string mapping the absolute file path to the final optimized weights file.
    """
    print("\n" + "="*60)
    print(" PHASE 2: SPECIALIZED FINE-TUNING ON SURGICAL INCISIONS")
    print("="*60)

    model = YOLO(model_path_phase1)
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        project="TFG_Incisions",
        name="phase2_incisions",
        save=True,
        plots=True,
        patience=30,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.8,
        degrees=180.0,
        scale=0.7,
        shear=10.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.2,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        dropout=0.1,
        weight_decay=0.0005,
    )
    return f"{results.save_dir}/weights/best.pt"


def independent_dataset_validation(best_model_phase2_path, yaml_path):
    """
    PRE-CONDITION:
        - `best_model_phase2_path` must map to the final optimized validation weights on disk.
        - `yaml_path` must target the matching validation config file declaring an independent test split.
    POST-CONDITION:
        - Validates system segmentation accuracy metrics against an unseen test configuration.
        - Outputs box and segmentation precision, recall, and mAP metrics directly to stdout.
        - Returns the underlying validation metrics object instance.
    """
    print("\n" + "="*60)
    print(" INDEPENDENT VALIDATION (EXTERNAL TEST SPLIT)")
    print("="*60)

    model = YOLO(best_model_phase2_path)
    metrics = model.val(data=yaml_path, split='test', device=0, project="TFG_Incisions", name="external_validation")

    print(f"\n REAL SEGMENTATION METRICS (EXTERNAL TEST):")
    print(f"  mAP 50-95 (Box):  {metrics.box.map:.4f}")
    print(f"  mAP 50 (Box):     {metrics.box.map50:.4f}")
    print(f"  mAP 50-95 (Seg):  {metrics.seg.map:.4f}")
    print(f"  mAP 50 (Seg):     {metrics.seg.map50:.4f}")
    print(f"  Precision (Seg):  {metrics.seg.mp:.4f}")
    print(f"  Recall (Seg):     {metrics.seg.mr:.4f}")

    return metrics


def main():
    """
    PRE-CONDITION:
        - Absolute dataset path configurations below must be updated to valid environment paths.
    POST-CONDITION:
        - Sequentially runs folder creation, general pre-training, fine-tuning, and evaluation blocks.
    """
    # =========================================================================
    # WORKSPACE PATH GENERALIZATION CONFIGURATION
    # =========================================================================
    DATASET_BASE_DIR = "path/to/Dataset"
    YAML_PHASE1_PATH = "path/to/general_wounds.yaml"
    YAML_PHASE2_PATH = "path/to/incisions.yaml"
    BASE_MODEL_WEIGHTS = "path/to/yolo26s-seg.pt"

    prepare_dir_structure(base_dir_placeholder=DATASET_BASE_DIR)
    
    model_phase1 = phase1_pretrain(BASE_MODEL_WEIGHTS, YAML_PHASE1_PATH)
    final_model = phase2_finetuning(model_phase1, YAML_PHASE2_PATH)
    
    independent_dataset_validation(final_model, YAML_PHASE2_PATH)


if __name__ == "__main__":
    main()