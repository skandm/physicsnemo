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
                        (default: surface_points_sample from config).
    predict.force_csv   Path to force.csv from the LBM simulation (optional).
                        If provided, the force-based L/D is computed by running
                        inference at the exact LBM cut-cell centers and summing
                        directly — no normalization ambiguity. Columns must be
                        x, y, z, fx, fy, fz (no header).

The script reads model / data config from conf/config.yaml via Hydra, and
loads the latest checkpoint found in cfg.resume_dir.
"""

import json
import math
from pathlib import Path

# Warp 1.x compatibility shim: physicsnemo uses wp.context.Device as a type
# annotation which was removed in warp 1.0. Must come before physicsnemo imports.
import warp as wp
import types as _types, sys as _sys
if not hasattr(wp, "context"):
    _ctx = _types.ModuleType("warp.context")
    _ctx.Device = object  # used as type annotation only, not at runtime
    wp.context = _ctx
    _sys.modules["warp.context"] = _ctx

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
        stl_mesh: triangulated pyvista PolyData with scaled coords and cell normals
                  (used for nearest-face projection in load_lbm_points)
    """
    stl = pv.read(str(stl_path))
    stl = stl.triangulate()

    verts_raw = np.array(stl.points, dtype=np.float32) * unit_scale
    stl_scaled = stl.copy()
    stl_scaled.points = verts_raw
    stl_scaled = stl_scaled.compute_normals(cell_normals=True, point_normals=False)

    verts   = verts_raw
    faces   = stl.faces.reshape(-1, 4)[:, 1:].flatten().astype(np.int32)
    centers = np.array(stl_scaled.cell_centers().points, dtype=np.float32)
    normals = np.array(stl_scaled.cell_data["Normals"], dtype=np.float32)

    sizes   = stl_scaled.compute_cell_sizes(length=False, area=True, volume=False)
    areas   = np.array(sizes.cell_data["Area"], dtype=np.float32)

    return verts, faces, centers, normals, areas, stl_scaled


# ──────────────────────────────────────────────────────────────────────────────
# LBM cut-cell loading
# ──────────────────────────────────────────────────────────────────────────────

def load_lbm_points(force_csv: Path, stl, unit_scale: float = 1.0):
    """Read LBM cut-cell centers from force.csv and project onto the STL.

    Projects each LBM point to the nearest STL face to inherit its normal
    and area — exactly the same procedure used in csv_to_zarr.py.

    Args:
        force_csv:  Path to force.csv (columns: x, y, z, fx, fy, fz).
        stl:        Triangulated pyvista PolyData with cell normals computed
                    and coordinates already scaled by unit_scale.
        unit_scale: Multiply CSV coordinates by this factor (mm → m).

    Returns:
        coords:  float32 [N_cells, 3] — LBM cell centers (scaled)
        normals: float32 [N_cells, 3] — nearest STL face normals
        areas:   float32 [N_cells]    — nearest STL face areas (scaled)
    """
    data = np.loadtxt(str(force_csv), delimiter=",", usecols=[0, 1, 2]).astype(np.float32)
    coords = data * unit_scale  # [N_cells, 3]

    # Project to nearest STL face for normals and areas
    sizes    = stl.compute_cell_sizes(length=False, area=True, volume=False)
    stl_areas = np.array(sizes.cell_data["Area"], dtype=np.float32) * (unit_scale ** 2)

    cell_ids, _ = stl.find_closest_cell(coords, return_closest_point=True)
    normals = stl.cell_data["Normals"][cell_ids].astype(np.float32)
    areas   = stl_areas[cell_ids].astype(np.float32)

    return coords, normals, areas


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
    stl_areas: torch.Tensor,
    global_params: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    batch_size: int,
    logger: PythonLogger,
    query_centers: torch.Tensor | None = None,
    query_normals: torch.Tensor | None = None,
    query_areas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DoMINO inference at a set of query points.

    The STL geometry arrays (stl_coordinates, stl_faces, stl_centers, stl_areas)
    always describe the full mesh and are used for geometry encoding.

    The query arrays (query_centers, query_normals, query_areas) define where
    predictions are made. If omitted they default to the STL face centers.

    Args:
        stl_coordinates: [N_verts, 3]       — STL vertex coordinates
        stl_faces:       [N_faces * 3]      — flat face index buffer
        stl_centers:     [N_faces, 3]       — STL face centroids (geometry context)
        stl_areas:       [N_faces]          — STL face areas (geometry context)
        global_params:   [n_params, 1]
        model:           Loaded DoMINO model (eval mode)
        datapipe:        Configured DoMINODataPipe
        batch_size:      Points per forward pass
        logger:          Logger instance
        query_centers:   [N_query, 3]  — points to predict at (default: stl_centers)
        query_normals:   [N_query, 3]  — normals at query points (default: stl normals)
        query_areas:     [N_query]     — areas at query points  (default: stl areas)

    Returns:
        preds: float32 [N_query, 4] = [pressure, force_x, force_y, force_z]
    """
    # Default query points to STL face centers
    if query_centers is None:
        query_centers = stl_centers
    if query_normals is None:
        query_normals = torch.zeros_like(query_centers)  # datapipe will handle missing normals
    if query_areas is None:
        query_areas = stl_areas

    n_query = query_centers.shape[0]

    if batch_size <= 0 or batch_size >= n_query:
        chunks = [(0, n_query)]
    else:
        n_chunks = math.ceil(n_query / batch_size)
        chunks = [(i * batch_size, min((i + 1) * batch_size, n_query)) for i in range(n_chunks)]

    logger.info(
        f"Running inference at {n_query} points "
        f"in {len(chunks)} chunk(s) of up to {batch_size if batch_size > 0 else n_query} points"
    )

    all_preds = []
    for i, (start, end) in enumerate(chunks):
        logger.info(f"  Chunk {i + 1}/{len(chunks)}: [{start}, {end})")

        inf_dict = _build_inference_dict(
            stl_coordinates = stl_coordinates,
            stl_faces       = stl_faces,
            stl_centers     = stl_centers,        # always full STL for geometry
            stl_areas       = stl_areas,           # always full STL for geometry
            global_params   = global_params,
            surface_centers = query_centers[start:end],
            surface_normals = query_normals[start:end],
            surface_areas   = query_areas[start:end],
        )

        preprocessed = datapipe.process_data(inf_dict)
        preprocessed = {k: v.unsqueeze(0) for k, v in preprocessed.items()}

        with torch.no_grad():
            _, output_surf = model(preprocessed)

        _, output_surf = datapipe.unscale_model_outputs(None, output_surf)
        all_preds.append(output_surf[0].cpu())

    return torch.cat(all_preds, dim=0)  # [N_query, 4]


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
    # Default to surface_points_sample to match the model's training batch size.
    # Resolved after cfg is loaded so we can reference cfg.model.surface_points_sample.
    batch_size   = int(predict_cfg.get("batch_size", -1))
    force_csv    = predict_cfg.get("force_csv", None)
    if force_csv is not None:
        force_csv = Path(force_csv)

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
    verts, faces_flat, centers_np, normals_np, areas_np, stl_mesh = load_stl(stl_path, stl_scale)

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

    # Resolve batch_size now that cfg is available.
    if batch_size < 0:
        batch_size = cfg.model.surface_points_sample

    datapipe = DoMINODataPipe(
        cfg.data.input_dir,              # path arg not used without a dataset
        phase                    = "test",
        grid_resolution          = cfg.model.interp_res,
        normalize_coordinates    = cfg.data.normalize_coordinates,
        sampling                 = False,   # we supply exact face centers; no subsampling
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
        stl_areas       = stl_areas_t,
        global_params   = global_params,
        model           = model,
        datapipe        = datapipe,
        batch_size      = batch_size,
        logger          = logger,
        query_centers   = stl_centers_t,
        query_normals   = stl_normals_t,
        query_areas     = stl_areas_t,
    )
    # preds: [N_faces, 4] = [pressure, force_x, force_y, force_z]

    pressure_arr = preds[:, 0].numpy()
    force_x_arr  = preds[:, 1].numpy()
    force_y_arr  = preds[:, 2].numpy()
    force_z_arr  = preds[:, 3].numpy()

    # ── L/D from pressure alone: F = -p * n * area ───────────────────────────
    # Pressure is a continuous field — summing over all STL faces is a
    # geometrically correct surface integral regardless of discretization.
    # normals_np: [N_faces, 3], areas_np: [N_faces]
    p_drag = float((-pressure_arr * normals_np[:, 0] * areas_np).sum())
    p_lift = float((-pressure_arr * normals_np[:, 2] * areas_np).sum())
    p_ld   = float(abs(p_lift) / (abs(p_drag) + 1e-8))

    logger.info(f"Pressure-based (inviscid): Lift={p_lift:.6g} N  Drag={p_drag:.6g} N  L/D={p_ld:.4f}")

    # ── Force-based L/D via area-weighted sampling ────────────────────────────
    # The model predicts force per LBM cut-cell (a discretization-dependent
    # quantity). Summing over all 511k STL faces over-counts by ~16x relative
    # to the ~32k training cells. To get the correct total, sample
    # surface_points_sample points area-weighted (replicating training scale)
    # and sum — this matches the scale the model was trained at.
    n_sample  = cfg.model.surface_points_sample
    probs     = (areas_np / areas_np.sum()).astype(np.float64)
    probs    /= probs.sum()   # ensure exact normalisation
    sample_idx = np.random.choice(len(centers_np), size=n_sample, replace=True, p=probs)

    force_inf_dict = _build_inference_dict(
        stl_coordinates = stl_coordinates,
        stl_faces       = stl_faces,
        stl_centers     = stl_centers_t,
        stl_areas       = stl_areas_t,
        global_params   = global_params,
        surface_centers = stl_centers_t[sample_idx],
        surface_normals = stl_normals_t[sample_idx],
        surface_areas   = stl_areas_t[sample_idx],
    )
    force_prep = datapipe.process_data(force_inf_dict)
    force_prep = {k: v.unsqueeze(0) for k, v in force_prep.items()}
    with torch.no_grad():
        _, force_out = model(force_prep)
    _, force_out = datapipe.unscale_model_outputs(None, force_out)
    # force_out: [1, n_sample, 4]

    drag = float(force_out[0, :, 1].sum().cpu())
    lift = float(force_out[0, :, -1].sum().cpu())
    ld   = float(abs(lift) / (abs(drag) + 1e-8))

    logger.info(f"Force-based  (N={n_sample} area-weighted): Lift={lift:.6g} N  Drag={drag:.6g} N  L/D={ld:.4f}")

    # ── LBM-exact force L/D (only if force_csv is provided) ──────────────────
    # Runs inference at the exact LBM cut-cell centers from the simulation.
    # Summing predictions over all N_cells is unambiguous — same discretization
    # as training, so no normalization factor needed.
    lbm_ld_dict = None
    if force_csv is not None:
        logger.info(f"Loading LBM cut-cell centers from {force_csv} ...")
        lbm_coords, lbm_normals, lbm_areas = load_lbm_points(force_csv, stl_mesh, stl_scale)
        n_lbm = lbm_coords.shape[0]
        logger.info(f"  {n_lbm} LBM cut-cells found")

        lbm_centers_t = torch.from_numpy(lbm_coords).to(device)
        lbm_normals_t = torch.from_numpy(lbm_normals).to(device)
        lbm_areas_t   = torch.from_numpy(lbm_areas).to(device)

        lbm_preds = run_inference(
            stl_coordinates = stl_coordinates,
            stl_faces       = stl_faces,
            stl_centers     = stl_centers_t,   # full STL for geometry context
            stl_areas       = stl_areas_t,     # full STL for geometry context
            global_params   = global_params,
            model           = model,
            datapipe        = datapipe,
            batch_size      = batch_size,
            logger          = logger,
            query_centers   = lbm_centers_t,   # predict at LBM cell centers
            query_normals   = lbm_normals_t,
            query_areas     = lbm_areas_t,
        )
        # lbm_preds: [N_cells, 4] = [pressure, force_x, force_y, force_z]

        lbm_drag = float(lbm_preds[:, 1].sum().cpu())
        lbm_lift = float(lbm_preds[:, -1].sum().cpu())
        lbm_ld   = float(abs(lbm_lift) / (abs(lbm_drag) + 1e-8))

        logger.info(f"LBM-exact (N={n_lbm}): Lift={lbm_lift:.6g} N  Drag={lbm_drag:.6g} N  L/D={lbm_ld:.4f}")

        lbm_ld_dict = {
            "note": f"Inference at exact LBM cut-cell centers (N={n_lbm}), direct sum",
            "lift": lbm_lift,
            "drag": lbm_drag,
            "ld":   lbm_ld,
        }

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
    ld_dict = {
        "force_sampled": {
            "note": f"Summed over {n_sample} area-weighted samples (matches training scale)",
            "lift": lift,
            "drag": drag,
            "ld":   ld,
        },
        "pressure": {
            "note": "Inviscid pressure integral over all STL faces; excludes viscous drag",
            "lift": p_lift,
            "drag": p_drag,
            "ld":   p_ld,
        },
    }
    if lbm_ld_dict is not None:
        ld_dict["force_lbm"] = lbm_ld_dict
    with open(output_dir / "l_d.json", "w") as fh:
        json.dump(ld_dict, fh, indent=2)

    logger.info(f"Outputs written to {output_dir}:")
    logger.info(f"  pressure.csv : {len(pressure_arr)} rows")
    logger.info(f"  force.csv    : {len(force_x_arr)} rows")
    logger.info(f"  l_d.json     : force L/D={ld:.4f}  pressure L/D={p_ld:.4f}")


if __name__ == "__main__":
    main()
