# Fork Changes Summary: dev/sk/ai vs. upstream/main

This document provides a high-level summary of the modifications introduced in the fork branch `dev/sk/ai` (authored by Skand) relative to the upstream base repository (`NVIDIA/physicsnemo:main`).

The changes focus on expanding and optimizing the **DoMINO** model, simplifying the **Crash** and **Stormcast** examples, removing experimental/deprecated architectures, and addressing multi-GPU stability.

---

## 1. DoMINO Model Enhancements & Hardening

The majority of development in this fork focuses on the DoMINO (Decomposable Multi-scale Iterative Neural Operator) external aerodynamics pipeline. The key areas of change include:

* **Zarr-Based Data Pipeline**: Transitioned data storage, shuffling, inspection, and splitting utilities from legacy formats to high-performance Zarr formats ([convert_to_zarr.py](file:///C:/cpp/physicsnemo/examples/cfd/external_aerodynamics/domino/src/convert_to_zarr.py), [csv_to_zarr.py](file:///C:/cpp/physicsnemo/examples/cfd/external_aerodynamics/domino/src/csv_to_zarr.py)). This allows memory-efficient streaming of large CFD simulation grids.
* **Direct-from-STL Inference**: Added a new end-to-end inference flow ([predict_from_stl.py](file:///C:/cpp/physicsnemo/examples/cfd/external_aerodynamics/domino/src/predict_from_stl.py)) that accepts raw CAD geometries (`.stl` files) and generates predicted surface pressure and shear forces directly.
* **Force-Based Training & Losses**: Aligned training supervision to target forces rather than velocities, using RMSE-based surface loss and masking padded points to fix scaling discrepancies.
* **Aerodynamic Metrics**: Integrated pressure-based and area-weighted Lift-to-Drag (L/D) metrics, fixing sign conventions and area biases.
* **Stability & OOM Prevention**:
  * Downsized the geometry encoder to prevent Out-Of-Memory (OOM) errors on modern GPUs (e.g., NVIDIA H100 NVL).
  * Adopted `bfloat16` precision for training to eliminate NaN instabilities during scale weight updates.
  * Batched final surface prediction passes to handle large meshes.
* **Optional Backend Resiliency**: Added check fallbacks so that the codebase remains functional in environments missing specific optional packages like RAPIDS `cuml`, NVIDIA `warp`, or advanced transformer layers.

---

## 2. Simplification & Deletion of Deprecated Models

To keep the repository maintainable, several experimental or unused features from the base repository were removed or simplified:

* **GLOBE Model Deletion**: Completely removed the experimental GLOBE (Global Local Operator Network) model codebase and its accompanying AirFRANS aerodynamics example.
* **Stormcast Refactoring**: Streamlined the Stormcast weather model training workflow, migrating it to standard `domain_parallel` APIs and purging obsolete mock configurations, dataset helpers, and parallel utilities.
* **Crash Model Simplification**: Restructured the crash simulation workflow (Part I) by introducing [d3plot_reader.py](file:///C:/cpp/physicsnemo/examples/structural_mechanics/crash/d3plot_reader.py) for LS-DYNA binary files and removing redundant utilities.
* **Core Library Cleanup**: Deleted deprecated layer implementations from the core `physicsnemo` package, including Pade activations, running normalization layers, and 3D equivariant operations.

---

## 3. Distributed Training & Infrastructure Improvements

* **Multi-GPU/DDP Memory Isolation**: Solved intermittent CUDA illegal memory access crashes during distributed (DDP) runs by enforcing GPU isolation per rank.
* **DDP Diagnostics**: Introduced a standalone multi-GPU verification script ([test_ddp.py](file:///C:/cpp/physicsnemo/test_ddp.py)) to debug communication backends independently of the full training code.
* **KNN Implementation Fixes**: Resolved edge-case crashes in PyTorch and cuML KNN search kernels when neighborhood queries exceed the total number of points.
