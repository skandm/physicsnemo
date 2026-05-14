# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DoMINO surface inference from a single STL file.

Loads a trained DoMINO surface model, runs predictions at every STL face
center, and writes three output files:

    <output_dir>/pressure.csv   — columns: x, y, z, pressure
    <output_dir>/force.csv      — columns: x, y, z, force_x, force_y, force_z
    <output_dir>/l_d.json       — {"lift": float, "drag": float, "ld": float}

Lift  = sum(force_z)   across all faces
Drag  = sum(force_x)   across all faces
L/D   = |Lift| / |Drag|

Usage (run from examples/cfd/external_aerodynamics/domino/src/):

    python predict_from_stl.py \\
        +predict.stl=/path/to/mesh.stl \\
        +predict.output_dir=/path/to/outputs \\
        +predict.stl_scale=0.001

Arguments (all passed as Hydra overrides):
    predict.stl         Path to the input STL file (required).
    predict.output_dir  Directory where CSV/JSON outputs are written (required).
    predict.stl_scale   Multiply STL coordinates by this factor before inference.
                        Use 0.001 if the STL is in mm and the model was trained
                        in meters (default: 0.001).
    predict.batch_size  Number of face centers processed per forward pass.
                        Reduce if you get CUDA OOM. 0 means process all at once
                        (default: 0).

The script reads model / data config from conf/config.yaml via Hydra, and
loads the latest checkpoint found in cfg.resume_dir.
"""

import json
import math
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import numpy as np
import pyvista as pv
import torch

# Configures cupy / PyTorch to share the GPU memory pool — must be imported early.
from physicsnemo.utils.memory import unified_gpu_memory  # noqa: F401

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO

from utils import load_scaling_factors, get_num_vars, coordinate_distributed_environment


# ──────────────────────────────────────────────────────────────────────────────
# STL loading
# ──────────────────────────────────────────────────────────────────────────────

def load_stl(stl_path: Path, unit_scale: float = 1.0):
    """Load an STL and return arrays needed for DoMINO inference.

    Args:
        stl_path:   Path to the STL file.
        unit_scale: Multiply all coordinates/areas by this factor.
                    Use 0.001 to convert mm → meters.

    Returns:
        verts:    float32 [N_verts, 3]  — vertex coordinates (scaled)
        faces:    int32   [N_faces * 3] — flattened face index buffer
        centers:  float32 [N_faces, 3] — face centroid coordinates (scaled)
        normals:  float32 [N_faces, 3] — outward unit normals
        areas:    float32 [N_faces]    — face areas (scaled by unit_scale²)
    """
    stl = pv.read(str(stl_path))
    stl = stl.triangulate()
    stl = stl.compute_normals(cell_normals=True, point_normals=False)

    verts   = np.array(stl.points, dtype=np.float32) * unit_scale
    faces   = stl.faces.reshape(-1, 4)[:, 1:].flatten().astype(np.int32)
    centers = np.array(stl.cell_centers().points, dtype=np.float32) * unit_scale
    normals = np.array(stl.cell_data["Normals"], dtype=np.float32)

    sizes   = stl.compute_cell_sizes(length=False, area=True, volume=False)
    areas   = np.array(sizes.cell_data["Area"], dtype=np.float32) * (unit_scale ** 2)

    return verts, faces, centers, normals, areas


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_inference_dict(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    stl_centers: torch.Tensor,
    stl_areas: torch.Tensor,
    global_params: torch.Tensor,
    surface_centers: torch.Tensor,
    surface_normals: torch.Tensor,
    surface_areas: torch.Tensor,
) -> dict:
    """Build the inference dictionary for one chunk of face centers."""
    return {
        "stl_coordinates":         stl_coordinates,
        "stl_faces":               stl_faces,
        "stl_centers":             stl_centers,
        "stl_areas":               stl_areas,
        "global_params_values":    global_params,
        "global_params_reference": global_params,
        "surface_mesh_centers":    surface_centers,
        "surface_normals":         surface_normals,
        "surface_areas":           surface_areas,
        "surface_faces":           stl_faces,
    }


def run_inference(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    stl_centers: torch.Tensor,
    stl_normals: torch.Tensor,
    stl_areas: torch.Tensor,
    global_params: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    batch_size: int,
    logger: PythonLogger,
) -> torch.Tensor:
    """Run DoMINO inference at all STL face centers.

    Args:
        stl_coordinates: [N_verts, 3]
        stl_faces:       [N_faces * 3]  — flat index buffer
        stl_centers:     [N_faces, 3]   — face centroids
        stl_normals:     [N_faces, 3]   — outward unit normals
        stl_areas:       [N_faces]      — face areas
        global_params:   [n_params, 1]
        model:           Loaded DoMINO model (in eval mode)
        datapipe:        Configured DoMINODataPipe (for preprocessing + unscaling)
        batch_size:      Chunk size for face-center processing (0 = all at once)
        logger:          Logger instance

    Returns:
        preds: float32 [N_faces, 4] = [pressure, force_x, force_y, force_z]
    """
    n_faces = stl_centers.shape[0]

    if batch_size <= 0 or batch_size >= n_faces:
        # Process all face centers in one forward pass.
        chunks = [(0, n_faces)]
    else:
        n_chunks = math.ceil(n_faces / batch_size)
        chunks = [(i * batch_size, min((i + 1) * batch_size, n_faces)) for i in range(n_chunks)]

    logger.info(
        f"Running inference at {n_faces} face centers "
        f"in {len(chunks)} chunk(s) of up to {batch_size if batch_size > 0 else n_faces} points"
    )

    all_preds = []
    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i + 1}/{len(chunks)}: faces [{start}, {end})")

        inf_dict = _build_inference_dict(
            stl_coordinates = stl_coordinates,
            stl_faces       = stl_faces,
            stl_centers     = stl_centers,
            stl_areas       = stl_areas,
            global_params   = global_params,
            surface_centers = stl_centers[start:end],
            surface_normals = stl_normals[start:end],
            surface_areas   = stl_areas[start:end],
        )

        preprocessed = datapipe.process_data(inf_dict)
        preprocessed = {k: v.unsqueeze(0) for k, v in preprocessed.items()}

        with torch.no_grad():
            _, output_surf = model(preprocessed)

        _, output_surf = datapipe.unscale_model_outputs(None, output_surf)
        # output_surf: [1, chunk_size, 4]
        all_preds.append(output_surf[0].cpu())

    return torch.cat(all_preds, dim=0)  # [N_faces, 4]


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ── Parse predict overrides ───────────────────────────────────────────────
    predict_cfg  = cfg.get("predict", {})
    stl_path     = Path(predict_cfg.get("stl"))
    output_dir   = Path(predict_cfg.get("output_dir"))
    stl_scale    = float(predict_cfg.get("stl_scale", 0.001))
    batch_size   = int(predict_cfg.get("batch_size", 0))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distributed / device setup ────────────────────────────────────────────
    DistributedManager.initialize()
    dist   = DistributedManager()
    device = dist.device

    nvmlInit()

    logger = PythonLogger("predict_from_stl")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"STL path:   {stl_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"STL scale:  {stl_scale}  (multiply coords/areas by this factor)")

    # ── Load STL geometry ─────────────────────────────────────────────────────
    logger.info("Loading STL...")
    verts, faces_flat, centers_np, normals_np, areas_np = load_stl(stl_path, stl_scale)

    logger.info(
        f"STL loaded: {verts.shape[0]} vertices, {centers_np.shape[0]} faces"
    )

    stl_coordinates = torch.from_numpy(verts).to(device)
    stl_faces       = torch.from_numpy(faces_flat).to(device)
    stl_centers_t   = torch.from_numpy(centers_np).to(device)
    stl_normals_t   = torch.from_numpy(normals_np).to(device)
    stl_areas_t     = torch.from_numpy(areas_np).to(device)

    # ── Global parameters from config ─────────────────────────────────────────
    params_vec = []
    for key in cfg.variables.global_parameters:
        p = cfg.variables.global_parameters[key]
        if p.type == "vector":
            params_vec.extend(list(p.reference))
        else:
            params_vec.append(float(p.reference))
    global_params = torch.tensor(params_vec, dtype=torch.float32, device=device).reshape(-1, 1)

    # ── Scaling factors ───────────────────────────────────────────────────────
    vol_factors, surf_factors = load_scaling_factors(cfg, logger)

    # ── Datapipe (no dataset — only used for process_data + unscale) ──────────
    domain_mesh, _, _ = coordinate_distributed_environment(cfg)

    overrides = {}
    if hasattr(cfg.data, "gpu_preprocessing"):
        overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing
    if hasattr(cfg.data, "gpu_output"):
        overrides["gpu_output"] = cfg.data.gpu_output

    datapipe = DoMINODataPipe(
        cfg.data.input_dir,              # path arg not used without a dataset
        phase                    = "test",
        grid_resolution          = cfg.model.interp_res,
        normalize_coordinates    = cfg.data.normalize_coordinates,
        sampling                 = cfg.data.sampling,
        sample_in_bbox           = cfg.data.sample_in_bbox,
        volume_points_sample     = cfg.model.volume_points_sample,
        surface_points_sample    = cfg.model.surface_points_sample,
        geom_points_sample       = cfg.model.geom_points_sample,
        volume_factors           = vol_factors,
        surface_factors          = surf_factors,
        scaling_type             = cfg.model.normalization,
        model_type               = cfg.model.model_type,
        bounding_box_dims        = cfg.data.bounding_box,
        bounding_box_dims_surf   = cfg.data.bounding_box_surface,
        volume_sample_from_disk  = cfg.data.volume_sample_from_disk,
        num_surface_neighbors    = cfg.model.num_neighbors_surface,
        surface_sampling_algorithm = cfg.model.surface_sampling_algorithm,
        **overrides,
    )

    # ── Build and load model ──────────────────────────────────────────────────
    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    model = DoMINO(
        input_features      = 3,
        output_features_vol = num_vol_vars,
        output_features_surf= num_surf_vars,
        global_features     = num_global_features,
        model_parameters    = cfg.model,
    ).to(device)

    load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models = model,
        device = device,
    )
    model.eval()
    logger.info("Checkpoint loaded.")

    # ── Inference ─────────────────────────────────────────────────────────────
    preds = run_inference(
        stl_coordinates = stl_coordinates,
        stl_faces       = stl_faces,
        stl_centers     = stl_centers_t,
        stl_normals     = stl_normals_t,
        stl_areas       = stl_areas_t,
        global_params   = global_params,
        model           = model,
        datapipe        = datapipe,
        batch_size      = batch_size,
        logger          = logger,
    )
    # preds: [N_faces, 4] = [pressure, force_x, force_y, force_z]

    pressure_arr = preds[:, 0].numpy()
    force_x_arr  = preds[:, 1].numpy()
    force_y_arr  = preds[:, 2].numpy()
    force_z_arr  = preds[:, 3].numpy()

    # ── L/D computation ───────────────────────────────────────────────────────
    drag = float(force_x_arr.sum())
    lift = float(force_z_arr.sum())
    ld   = float(abs(lift) / (abs(drag) + 1e-8))

    logger.info(f"Lift  = {lift:.6g} N")
    logger.info(f"Drag  = {drag:.6g} N")
    logger.info(f"L/D   = {ld:.4f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    # pressure.csv: x, y, z, pressure
    p_out = np.column_stack([centers_np, pressure_arr])
    np.savetxt(
        str(output_dir / "pressure.csv"),
        p_out,
        delimiter = ",",
        header    = "x,y,z,pressure",
        comments  = "",
    )

    # force.csv: x, y, z, force_x, force_y, force_z
    f_out = np.column_stack([centers_np, force_x_arr, force_y_arr, force_z_arr])
    np.savetxt(
        str(output_dir / "force.csv"),
        f_out,
        delimiter = ",",
        header    = "x,y,z,force_x,force_y,force_z",
        comments  = "",
    )

    # l_d.json
    ld_dict = {"lift": lift, "drag": drag, "ld": ld}
    with open(output_dir / "l_d.json", "w") as fh:
        json.dump(ld_dict, fh, indent=2)

    logger.info(f"Outputs written to {output_dir}:")
    logger.info(f"  pressure.csv : {len(pressure_arr)} rows")
    logger.info(f"  force.csv    : {len(force_x_arr)} rows")
    logger.info(f"  l_d.json     : {json.dumps(ld_dict)}")


if __name__ == "__main__":
    main()
