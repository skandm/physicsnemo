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
Standalone DoMINO inference script for volume-only predictions.

Given a new STL geometry, predicts [Vx, Vy, Vz, P] at random points
throughout the bounding box and saves results as a VTU file for ParaView.

Usage:
    # Minimal — config and scaling path are both read from the training run:
    python run_inference.py \\
        --stl        /path/to/aircraft.stl \\
        --checkpoint /path/to/outputs/RAF_CFD/1/models

    # Full options:
    python run_inference.py \\
        --stl        /path/to/aircraft.stl \\
        --checkpoint /path/to/outputs/RAF_CFD/1/models \\
        --scaling    /path/to/scaling_factors.pkl \\   # optional override
        --config     conf/config.yaml \\               # optional override
        --output     predicted_volume.vtu \\
        --num_points 500000

The script will:
  1. Load the STL and convert to GPU tensors
  2. Set up the DoMINODataPipe for preprocessing (no dataset required)
  3. Load the trained DoMINO model from the latest checkpoint
  4. Sample `num_points` random points in the bounding box in batches
  5. Run the model and unscale the outputs
  6. Save a VTU point cloud with Vx, Vy, Vz, P (and |V|) as arrays
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Shared-memory GPU pool for cupy/pytorch:
from physicsnemo.utils.memory import unified_gpu_memory  # noqa: F401  (side-effect import)

from omegaconf import DictConfig, OmegaConf

import logging as _logging

from physicsnemo.distributed import DistributedManager


def _make_logger():
    log = _logging.getLogger("run_inference")
    log.setLevel(_logging.INFO)
    if not log.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(
            _logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        log.addHandler(handler)
    return log

from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.geometry_rep import scale_sdf

from utils import get_num_vars, load_scaling_factors


# ---------------------------------------------------------------------------
# Helpers copied / adapted from inference_on_stl.py
# ---------------------------------------------------------------------------

def load_stl_to_tensors(stl_path: str, device: torch.device):
    """
    Read an STL file and return (stl_coordinates, stl_faces) as float32/int32
    tensors on `device`.

    Requires pyvista (pip install pyvista).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "pyvista is required to load STL files.  "
            "Install with:  pip install pyvista"
        )

    mesh = pv.read(stl_path)
    # pyvista stores faces as [n_verts_per_face, v0, v1, v2, ...] for each cell.
    # For a pure triangle mesh, every face is 4 ints: [3, i, j, k]
    faces_raw = mesh.faces
    n_faces = mesh.n_cells

    if faces_raw[0] != 3:
        # Triangulate if needed
        mesh = mesh.triangulate()
        faces_raw = mesh.faces
        n_faces = mesh.n_cells

    # Reshape and strip the leading "3":
    faces = faces_raw.reshape(n_faces, 4)[:, 1:]   # [N_faces, 3]

    stl_coordinates = torch.tensor(
        np.array(mesh.points, dtype=np.float32), device=device
    )
    stl_faces = torch.tensor(
        faces.astype(np.int32).flatten(), device=device, dtype=torch.int32
    )

    return stl_coordinates, stl_faces


def compute_stl_centers_and_areas(
    stl_coordinates: torch.Tensor, stl_faces: torch.Tensor
):
    """Compute triangle centroids and areas from vertices and face indices."""
    n_faces = stl_faces.shape[0] // 3
    tri_verts = stl_coordinates[stl_faces.reshape(-1, 3)]  # [N_faces, 3, 3]
    stl_centers = tri_verts.mean(dim=1)                    # [N_faces, 3]

    d1 = tri_verts[:, 1] - tri_verts[:, 0]
    d2 = tri_verts[:, 2] - tri_verts[:, 0]
    cross = torch.linalg.cross(d1, d2, dim=1)
    normals_norm = torch.linalg.norm(cross, dim=1)
    stl_areas = 0.5 * normals_norm                         # [N_faces]

    return stl_centers, stl_areas


def sample_volume_points(
    c_min: torch.Tensor,
    c_max: torch.Tensor,
    n_points: int,
    device: torch.device,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Uniform random sampling inside a bounding box."""
    uniform = (
        torch.rand(n_points, 3, device=device, dtype=torch.float32) * (1 - 2 * eps)
        + eps
    )
    return (c_max - c_min) * uniform + c_min


def reject_interior_points(preprocessed_data: dict) -> dict:
    """Remove volume points inside the solid geometry.

    Warp's signed_distance_field with use_sign_winding_number=True uses the
    convention: positive SDF = OUTSIDE (fluid), negative SDF = INSIDE the mesh.
    We keep only the exterior/fluid points (sdf > 0).
    """
    valid = preprocessed_data["sdf_nodes"].squeeze(-1) > 0
    for key in [
        "volume_mesh_centers",
        "sdf_nodes",
        "pos_volume_closest",
        "pos_volume_center_of_mass",
    ]:
        if key in preprocessed_data:
            preprocessed_data[key] = preprocessed_data[key][valid]
    return preprocessed_data


# ---------------------------------------------------------------------------
# Geometry encoding cache helpers
# ---------------------------------------------------------------------------

def _compute_geo_encoding(data_batched: dict, model: DoMINO) -> torch.Tensor:
    """
    Run geo_rep_volume once for a fixed geometry and return the latent
    geometry feature grid (encoding_g_vol).

    This is the expensive 3D-CNN stage of the DoMINO forward pass.  It
    depends only on the STL geometry (geometry_coordinates, sdf_grid, grid)
    and NOT on the query-point locations, so it can be cached across batches.
    """
    geo_centers = data_batched["geometry_coordinates"]  # [1, N_geo, 3]

    # Replicate exactly the geo-center normalisation from DoMINO.forward():
    # vol_min_max[:, 1] stores c_max and [:, 0] stores c_min per the datapipe
    # convention — the model indexes them this way, so we must too.
    if "volume_min_max" in data_batched:
        vol_min = data_batched["volume_min_max"][:, 1, :]  # [1, 3]
        vol_max = data_batched["volume_min_max"][:, 0, :]  # [1, 3]
        geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1.0
    else:
        geo_centers_vol = geo_centers

    p_grid   = data_batched["grid"]      # latent space grid  [1, ...]
    sdf_grid = data_batched["sdf_grid"]  # SDF on latent grid [1, ...]

    with torch.no_grad():
        encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

    return encoding_g_vol


def _decode_volume_batch(
    data_batched: dict,
    encoding_g_vol: torch.Tensor,
    model: DoMINO,
) -> torch.Tensor:
    """
    Run the per-batch portion of DoMINO.forward() for volume predictions.

    Skips geo_rep_volume (already cached in encoding_g_vol) and starts from
    volume_local_geo_encodings — which DOES depend on the per-batch query
    point positions (volume_mesh_centers).

    Mirrors lines 651-663 of DoMINO.forward() exactly.
    """
    volume_mesh_centers = data_batched["volume_mesh_centers"]   # [1, N_pts, 3]
    p_grid              = data_batched["grid"]                   # [1, ...]

    # 1. Interpolate latent geometry features at each query point position.
    encoding_g_vol_local = model.volume_local_geo_encodings(
        0.5 * encoding_g_vol, volume_mesh_centers, p_grid
    )

    # 2. Positional / SDF encoding (mirrors lines 589-615 of DoMINO.forward()).
    sdf_nodes = data_batched["sdf_nodes"]
    if model.use_sdf_in_basis_func:
        scaled_sdf = [scale_sdf(sdf_nodes, s) for s in model.sdf_scaling_factor]
        scaled_sdf = torch.cat(scaled_sdf, dim=-1)
        encoding_node_vol = torch.cat(
            (
                sdf_nodes,
                scaled_sdf,
                data_batched["pos_volume_closest"],
                data_batched["pos_volume_center_of_mass"],
            ),
            dim=-1,
        )
    else:
        encoding_node_vol = data_batched["pos_volume_center_of_mass"]

    encoding_node_vol = model.fc_p_vol(encoding_node_vol)

    # 3. Solution calculation.
    output_vol = model.solution_calculator_vol(
        volume_mesh_centers,
        encoding_g_vol_local,
        encoding_node_vol,
        data_batched["global_params_values"],
        data_batched["global_params_reference"],
    )

    return output_vol


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

def run_inference(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    batch_size: int,
    total_points: int,
    logger,
):
    """
    Predict volume fields at `total_points` locations in the bounding box.

    Returns:
        volume_coords  — [M, 3]  physical (metres) coordinates of accepted exterior points
        volume_preds   — [M, C]  predicted fields (unnormalised, physical units)
    """
    device = stl_coordinates.device

    stl_centers, stl_areas = compute_stl_centers_and_areas(stl_coordinates, stl_faces)

    c_min = datapipe.config.bounding_box_dims[1]   # [3]  lower corner
    c_max = datapipe.config.bounding_box_dims[0]   # [3]  upper corner
    c_min_cpu = c_min.cpu()
    c_max_cpu = c_max.cpu()

    # -------------------------------------------------------------------
    # One-time static preprocessing: SDF surface grid + geometry coords.
    # These depend only on the STL geometry, which never changes between
    # batches.  Computing them once and injecting the cache cuts per-batch
    # time from ~1-3 s down to ~200-500 ms.
    # -------------------------------------------------------------------
    logger.info("Computing static preprocessing (SDF surface grid) — one-time cost …")
    t_static = time.perf_counter()
    _seed_dict = {
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "stl_centers": stl_centers,
        "stl_areas": stl_areas,
        "global_params_values": global_params_values,
        "global_params_reference": global_params_reference,
        "volume_mesh_centers": sample_volume_points(c_min, c_max, batch_size, device),
    }
    _seed_result = datapipe.process_data(_seed_dict)
    static_cache = {
        "sdf_surf_grid":        _seed_result["sdf_surf_grid"],
        "surf_grid":             _seed_result["surf_grid"],
        "geometry_coordinates":  _seed_result["geometry_coordinates"],
        "sdf_grid":              _seed_result["sdf_grid"],
        "grid":                  _seed_result["grid"],
    }
    logger.info(f"Static preprocessing done in {time.perf_counter() - t_static:.1f}s")

    # -------------------------------------------------------------------
    # One-time geometry encoding (geo_rep_volume 3D CNN).
    # This is the most expensive model stage and depends only on the STL,
    # so we run it once and reuse the result for every subsequent batch.
    # -------------------------------------------------------------------
    logger.info("Computing geometry encoding (one-time model stage) …")
    t_geo = time.perf_counter()
    _geo_input = {k: v.unsqueeze(0) for k, v in _seed_result.items()}
    encoding_g_vol = _compute_geo_encoding(_geo_input, model)
    logger.info(f"Geometry encoding done in {time.perf_counter() - t_geo:.1f}s")

    all_coords = []
    all_preds = []
    # Collect the first batch result too — it was computed as part of the seed:
    _seed_result = reject_interior_points(_seed_result)
    if _seed_result["volume_mesh_centers"].shape[0] > 0:
        coords_norm = _seed_result["volume_mesh_centers"].cpu()
        coords_phys = coords_norm * (c_max_cpu - c_min_cpu) + c_min_cpu if datapipe.config.normalize_coordinates else coords_norm
        _seed_batched = {k: v.unsqueeze(0) for k, v in _seed_result.items()}
        with torch.no_grad():
            _out_vol = _decode_volume_batch(_seed_batched, encoding_g_vol, model)
        _out_vol, _ = datapipe.unscale_model_outputs(volume_fields=_out_vol)
        all_coords.append(coords_phys)
        all_preds.append(_out_vol.squeeze(0).cpu())

    total_accepted = sum(c.shape[0] for c in all_coords)
    total_queried = batch_size
    t0 = time.perf_counter()

    batch_idx = 1
    while total_accepted < total_points:
        t_batch = time.perf_counter()

        logger.info(f"  Batch {batch_idx}: sampling {batch_size:,} points …")
        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
            "volume_mesh_centers": sample_volume_points(
                c_min, c_max, batch_size, device
            ),
            **static_cache,  # inject pre-computed SDF grid + geometry coords
        }

        logger.info(f"  Batch {batch_idx}: preprocessing …")
        preprocessed = datapipe.process_data(inference_dict)
        preprocessed = reject_interior_points(preprocessed)

        n_accepted = preprocessed["volume_mesh_centers"].shape[0]
        if n_accepted == 0:
            logger.info(f"  Batch {batch_idx}: all points inside geometry, resampling …")
            continue

        # Extract coordinates BEFORE adding batch dim.
        # If normalize_coordinates=True the datapipe returns normalised [0,1] coords,
        # so we convert back to physical metres for the VTU file.
        coords_norm = preprocessed["volume_mesh_centers"].cpu()
        if datapipe.config.normalize_coordinates:
            coords_phys = coords_norm * (c_max_cpu - c_min_cpu) + c_min_cpu
        else:
            coords_phys = coords_norm

        preprocessed = {k: v.unsqueeze(0) for k, v in preprocessed.items()}

        logger.info(f"  Batch {batch_idx}: model forward ({n_accepted:,} exterior pts) …")
        with torch.no_grad():
            output_vol = _decode_volume_batch(preprocessed, encoding_g_vol, model)

        output_vol, _ = datapipe.unscale_model_outputs(volume_fields=output_vol)

        all_coords.append(coords_phys)
        all_preds.append(output_vol.squeeze(0).cpu())

        total_accepted += n_accepted
        total_queried += batch_size
        batch_time = time.perf_counter() - t_batch
        elapsed = time.perf_counter() - t0

        logger.info(
            f"  Batch {batch_idx} done: "
            f"{n_accepted:,} accepted ({batch_size - n_accepted:,} inside geometry), "
            f"{batch_time:.1f}s/batch — "
            f"total {total_accepted:,}/{total_points:,} pts ({elapsed:.1f}s elapsed)"
        )
        batch_idx += 1

    volume_coords = torch.cat(all_coords, dim=0)[:total_points]
    volume_preds = torch.cat(all_preds, dim=0)[:total_points]

    return volume_coords.numpy(), volume_preds.numpy()


# ---------------------------------------------------------------------------
# Grid-based inference (for VTI output)
# ---------------------------------------------------------------------------

def run_inference_grid(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    global_params_values: torch.Tensor,
    global_params_reference: torch.Tensor,
    model: "DoMINO",
    datapipe: "DoMINODataPipe",
    resolution: tuple[int, int, int],
    batch_size: int,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    Evaluate the model at every cell centre of a regular nx×ny×nz grid.

    Points inside the solid geometry (SDF ≤ 0) are skipped; their cells
    are left as zero in the output.  No interpolation is involved.

    Returns
    -------
    preds_flat : [nx*ny*nz, C]  predictions in VTK cell order (x varies fastest)
    bbox_min   : [3]  physical lower corner
    bbox_max   : [3]  physical upper corner
    spacing    : (dx, dy, dz)  cell size in metres
    """
    nx, ny, nz = resolution
    device = stl_coordinates.device

    c_min = datapipe.config.bounding_box_dims[1]   # lower corner tensor
    c_max = datapipe.config.bounding_box_dims[0]   # upper corner tensor
    bbox_min = c_min.cpu().numpy()
    bbox_max = c_max.cpu().numpy()

    dx = float((c_max[0] - c_min[0]) / nx)
    dy = float((c_max[1] - c_min[1]) / ny)
    dz = float((c_max[2] - c_min[2]) / nz)

    # Cell centres in VTK order: x varies fastest (inner), z slowest (outer).
    # all_points[k*ny*nx + j*nx + i] = centre of cell (i, j, k).
    xs = c_min[0] + (torch.arange(nx, device=device, dtype=torch.float32) + 0.5) * dx
    ys = c_min[1] + (torch.arange(ny, device=device, dtype=torch.float32) + 0.5) * dy
    zs = c_min[2] + (torch.arange(nz, device=device, dtype=torch.float32) + 0.5) * dz
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")  # each [nz, ny, nx]
    all_points = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1
    )  # [nx*ny*nz, 3]

    total_cells = nx * ny * nz
    n_batches = -(-total_cells // batch_size)
    logger.info(
        f"  Grid {nx}×{ny}×{nz} = {total_cells:,} cells, "
        f"spacing=({dx:.4f}, {dy:.4f}, {dz:.4f}) m, {n_batches} batches"
    )

    stl_centers, stl_areas = compute_stl_centers_and_areas(stl_coordinates, stl_faces)

    # -----------------------------------------------------------------------
    # One-time static preprocessing (SDF surface + volume grids)
    # -----------------------------------------------------------------------
    logger.info("  Computing static preprocessing (one-time) …")
    t_static = time.perf_counter()
    _seed_dict = {
        "stl_coordinates": stl_coordinates,
        "stl_faces":        stl_faces,
        "stl_centers":      stl_centers,
        "stl_areas":        stl_areas,
        "global_params_values":     global_params_values,
        "global_params_reference":  global_params_reference,
        "volume_mesh_centers": all_points[:batch_size],
    }
    _seed_result = datapipe.process_data(_seed_dict)
    static_cache = {
        "sdf_surf_grid":       _seed_result["sdf_surf_grid"],
        "surf_grid":            _seed_result["surf_grid"],
        "geometry_coordinates": _seed_result["geometry_coordinates"],
        "sdf_grid":             _seed_result["sdf_grid"],
        "grid":                 _seed_result["grid"],
    }
    logger.info(f"  Static preprocessing done in {time.perf_counter() - t_static:.1f}s")

    # -----------------------------------------------------------------------
    # One-time geometry encoding (geo_rep_volume 3D CNN)
    # -----------------------------------------------------------------------
    logger.info("  Computing geometry encoding (one-time model stage) …")
    t_geo = time.perf_counter()
    _geo_input = {k: v.unsqueeze(0) for k, v in _seed_result.items()}
    encoding_g_vol = _compute_geo_encoding(_geo_input, model)
    logger.info(f"  Geometry encoding done in {time.perf_counter() - t_geo:.1f}s")

    # Output buffer: zero for interior/skipped cells
    # n_channels discovered after first successful batch
    preds_flat = None
    all_flat_idx = []
    all_batch_preds = []

    def _process_batch(batch_points, batch_start, extra_dict):
        preprocessed = datapipe.process_data(extra_dict)
        sdf = preprocessed["sdf_nodes"].squeeze(-1)
        ext_mask = sdf > 0
        n_ext = int(ext_mask.sum())
        if n_ext == 0:
            return None, None
        n_pts = sdf.shape[0]
        ext = {
            k: v[ext_mask] if (v.ndim > 0 and v.shape[0] == n_pts) else v
            for k, v in preprocessed.items()
        }
        ext_batched = {k: v.unsqueeze(0) for k, v in ext.items()}
        with torch.no_grad():
            out_vol = _decode_volume_batch(ext_batched, encoding_g_vol, model)
        out_vol, _ = datapipe.unscale_model_outputs(volume_fields=out_vol)
        flat_idx = torch.arange(batch_start, batch_start + len(batch_points),
                                device=device)[ext_mask].cpu().numpy()
        return flat_idx, out_vol.squeeze(0).cpu().numpy()

    # Seed batch
    seed_idx, seed_preds = _process_batch(
        all_points[:batch_size], 0,
        {**_seed_dict}  # already has volume_mesh_centers = first batch
    )
    if seed_idx is not None:
        all_flat_idx.append(seed_idx)
        all_batch_preds.append(seed_preds)

    t0 = time.perf_counter()
    for batch_num, batch_start in enumerate(range(batch_size, total_cells, batch_size), start=1):
        batch_end = min(batch_start + batch_size, total_cells)
        batch_points = all_points[batch_start:batch_end]

        inf_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces":        stl_faces,
            "stl_centers":      stl_centers,
            "stl_areas":        stl_areas,
            "global_params_values":    global_params_values,
            "global_params_reference": global_params_reference,
            "volume_mesh_centers": batch_points,
            **static_cache,
        }
        flat_idx, batch_preds = _process_batch(batch_points, batch_start, inf_dict)
        if flat_idx is not None:
            all_flat_idx.append(flat_idx)
            all_batch_preds.append(batch_preds)

        if batch_num % 10 == 0 or batch_end == total_cells:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  Batch {batch_num}/{n_batches - 1}: "
                f"{batch_end:,}/{total_cells:,} cells processed ({elapsed:.1f}s)"
            )

    # Scatter into flat output array
    if all_batch_preds:
        n_channels = all_batch_preds[0].shape[1]
        preds_flat = np.zeros((total_cells, n_channels), dtype=np.float32)
        flat_idx_all = np.concatenate(all_flat_idx)
        preds_all    = np.concatenate(all_batch_preds, axis=0)
        preds_flat[flat_idx_all] = preds_all
    else:
        preds_flat = np.zeros((total_cells, 1), dtype=np.float32)

    return preds_flat, bbox_min, bbox_max, (dx, dy, dz)


# ---------------------------------------------------------------------------
# VTU output
# ---------------------------------------------------------------------------

def _build_cloud(coords: np.ndarray, preds: np.ndarray, variable_names: list[str]):
    """Create a pyvista PolyData point cloud with all predicted fields attached."""
    import pyvista as pv
    cloud = pv.PolyData(coords.astype(np.float64))
    for col, name in enumerate(variable_names):
        cloud.point_data[name] = preds[:, col]
    if preds.shape[1] >= 3:
        cloud.point_data["velocity_magnitude"] = np.linalg.norm(preds[:, :3], axis=1)
    return cloud


def save_vtu(
    coords: np.ndarray,
    preds: np.ndarray,
    output_path: str,
    variable_names: list[str],
):
    """
    Save a point cloud as a VTU/VTP file.

    NOTE: a bare point cloud cannot be sliced in ParaView.
    Use save_vti() for slice / volume-rendering support.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required:  pip install pyvista")

    cloud = _build_cloud(coords, preds, variable_names)

    ext = Path(output_path).suffix.lower()
    if ext == ".vtu":
        cloud.cast_to_unstructured_grid().save(output_path)
    else:
        cloud.save(output_path)
    print(f"Saved {coords.shape[0]:,} points to: {output_path}")


def save_vti_direct(
    preds_flat: np.ndarray,
    output_path: str,
    variable_names: list[str],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: tuple[int, int, int],
    output_names: dict[str, str] | None = None,
):
    """
    Save model predictions evaluated directly on a regular grid as VTI.

    The predictions were computed at cell centers so no interpolation is needed.
    Interior cells (inside the solid geometry) have been left as zero.

    VTI fully supports ParaView slice, threshold, volume rendering, streamlines.

    Args:
        variable_names: Per-column names as built from config (e.g. ``['U_time_avg_x',
                        'U_time_avg_y', 'U_time_avg_z', 'p_time_avg']``).  Consecutive
                        ``_x / _y / _z`` triplets are automatically stacked into a single
                        3-component vector array so ParaView can display streamlines and
                        glyphs without an extra ``Calculator`` step.
        output_names:   Optional mapping from the base variable name (the part before
                        ``_x/_y/_z`` for vectors, or the plain name for scalars) to the
                        desired array name in the output file.
                        Example: ``{'U_time_avg': 'velocity_time_avg',
                                    'p_time_avg': 'pressure_time_avg'}``
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required:  pip install pyvista")

    nx, ny, nz = resolution
    dx = (bbox_max[0] - bbox_min[0]) / nx
    dy = (bbox_max[1] - bbox_min[1]) / ny
    dz = (bbox_max[2] - bbox_min[2]) / nz

    # ImageData: dimensions = number of NODES = cells + 1 in each direction.
    # Cell data is indexed in VTK order: x varies fastest, z slowest.
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.origin  = (float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]))
    grid.spacing = (float(dx), float(dy), float(dz))

    rename = output_names or {}

    # Walk variable_names, grouping consecutive _x/_y/_z triplets into vectors.
    i = 0
    while i < len(variable_names):
        name = variable_names[i]
        if (
            name.endswith("_x")
            and i + 2 < len(variable_names)
            and variable_names[i + 1] == name[:-1] + "y"
            and variable_names[i + 2] == name[:-1] + "z"
        ):
            base = name[:-2]  # strip "_x"
            out_name = rename.get(base, base)
            vec = preds_flat[:, i : i + 3].astype(np.float32)
            grid.cell_data[out_name] = vec  # [N, 3] — ParaView treats as vector
            i += 3
        else:
            out_name = rename.get(name, name)
            grid.cell_data[out_name] = preds_flat[:, i].astype(np.float32)
            i += 1

    # ImplicitField: scalar sentinel filled with -1.0, used by downstream tools
    # to identify the fluid domain (interior solid cells remain at their zero
    # prediction values and can be masked by this field).
    grid.cell_data["ImplicitField"] = np.full(nx * ny * nz, -1.0, dtype=np.float32)

    # Convert cell data to point data so ParaView can interpolate smoothly
    # across voxel boundaries (required for streamlines, smooth iso-surfaces, etc.).
    grid = grid.cell_data_to_point_data(pass_cell_data=False)

    grid.save(output_path)
    print(f"Saved {nx}×{ny}×{nz} VTI to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_datapipe(
    cfg: DictConfig,
    vol_factors: torch.Tensor,
    sampling: bool | None = None,
) -> DoMINODataPipe:
    """
    Build a DoMINODataPipe for inference only (no dataset, no iteration).
    We only use it for process_data() and unscale_model_outputs().

    Parameters
    ----------
    sampling : override the config sampling flag.
        Pass False for grid-based inference so the datapipe processes every
        input point exactly once without random subsampling.
    """
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    overrides = {}
    if hasattr(cfg.data, "gpu_preprocessing"):
        overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing
    if hasattr(cfg.data, "gpu_output"):
        overrides["gpu_output"] = cfg.data.gpu_output

    datapipe = DoMINODataPipe(
        input_path=None,  # No dataset — we provide data_dict manually
        phase="test",
        model_type=cfg.model.model_type,
        grid_resolution=cfg.model.interp_res,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sampling=cfg.data.sampling if sampling is None else sampling,
        sample_in_bbox=cfg.data.sample_in_bbox,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        volume_factors=vol_factors,
        surface_factors=None,
        scaling_type=cfg.model.normalization,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        volume_sample_from_disk=False,  # No disk sampling for inference
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        **overrides,
    )

    return datapipe


def parse_args():
    p = argparse.ArgumentParser(
        description="Run DoMINO volume inference on a new STL geometry."
    )
    p.add_argument(
        "--stl",
        required=True,
        help="Path to input STL file (aircraft geometry)",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Directory containing saved .pt checkpoints (e.g. outputs/RAF_CFD/1/models)",
    )
    p.add_argument(
        "--scaling",
        default=None,
        help=(
            "Path to scaling_factors.pkl.  "
            "If omitted, uses data.scaling_factors from the config."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Path to config.yaml.  "
            "If omitted, the script looks for the Hydra-saved config at "
            "<checkpoint_dir>/../hydra/config.yaml (written automatically by train.py), "
            "then falls back to conf/config.yaml in the current directory."
        ),
    )
    p.add_argument(
        "--output",
        default="predicted_volume.vtu",
        help="Output VTU file path (default: predicted_volume.vtu)",
    )
    p.add_argument(
        "--num_points",
        type=int,
        default=500_000,
        help="Number of exterior volume points to predict (default: 500000)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Points per inference batch (default: volume_points_sample from config)",
    )
    p.add_argument(
        "--inlet_velocity",
        type=float,
        default=None,
        help="Override inlet velocity magnitude (m/s). Uses config default if not set.",
    )
    p.add_argument(
        "--air_density",
        type=float,
        default=None,
        help="Override air density (kg/m³). Uses config default if not set.",
    )
    p.add_argument(
        "--vti_resolution",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help=(
            "Grid resolution for VTI output (nx ny nz). "
            "Only used when --output ends in .vti. "
            "Defaults to model.interp_res from config (e.g. 128 64 64)."
        ),
    )
    return p.parse_args()


def _step(logger, n, total, msg):
    logger.info(f"[{n}/{total}] {msg}")


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Initialise distributed (single-GPU path)
    # -----------------------------------------------------------------------
    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device

    logger = _make_logger()
    logger.info("=" * 60)
    logger.info("DoMINO Volume Inference")
    logger.info("=" * 60)
    logger.info(f"Device : {device}")
    logger.info(f"STL    : {args.stl}")
    logger.info(f"Output : {args.output}")
    logger.info(f"Points : {args.num_points:,}")
    logger.info("=" * 60)

    N_STEPS = 6

    # -----------------------------------------------------------------------
    # Load config
    # Resolution order:
    #   1. --config <path>  (explicit)
    #   2. <checkpoint_dir>/../hydra/config.yaml  (saved by train.py automatically)
    #   3. conf/config.yaml in the current working directory  (fallback)
    # -----------------------------------------------------------------------
    checkpoint_dir = Path(args.checkpoint)
    hydra_saved = checkpoint_dir.parent / "hydra" / "config.yaml"

    if args.config is not None:
        config_path = Path(args.config).resolve()
    elif hydra_saved.exists():
        config_path = hydra_saved.resolve()
        logger.info(f"Auto-detected config from training run: {config_path}")
    else:
        config_path = Path("conf/config.yaml").resolve()
        logger.info(f"Falling back to local config: {config_path}")

    if not config_path.exists():
        logger.error(
            f"Config not found: {config_path}\n"
            f"Pass --config /path/to/config.yaml explicitly."
        )
        sys.exit(1)

    # OmegaConf.load handles both the original conf/config.yaml (resolves
    # ${...} interpolations lazily) and the Hydra-saved config (already fully
    # resolved).  No hydra initialisation needed either way.
    cfg = OmegaConf.load(config_path)

    # If --scaling was given, override the path from config:
    if args.scaling is not None:
        OmegaConf.update(cfg, "data.scaling_factors", args.scaling, merge=True)

    scaling_path = cfg.data.scaling_factors
    if not Path(scaling_path).exists():
        logger.error(
            f"Scaling factors not found: {scaling_path}\n"
            f"Pass --scaling /path/to/scaling_factors.pkl to override."
        )
        sys.exit(1)

    _step(logger, 1, N_STEPS, f"Config loaded from: {config_path}")
    logger.info(f"         Scaling factors: {scaling_path}")

    # -----------------------------------------------------------------------
    # Load scaling factors
    # -----------------------------------------------------------------------
    _step(logger, 2, N_STEPS, "Loading scaling factors …")
    vol_factors, surf_factors = load_scaling_factors(cfg)
    logger.info(f"         vol_factors shape: {vol_factors.shape}")

    # -----------------------------------------------------------------------
    # Build global params tensors
    # -----------------------------------------------------------------------
    gp = cfg.variables.global_parameters
    gp_vals = []
    gp_refs = []
    stream_velocity = 1.0   # for two-stage denormalization
    air_density = 1.0       # for two-stage denormalization

    for param_name, param_cfg in gp.items():
        if param_cfg.type == "vector":
            vals = list(param_cfg.reference)
        else:
            vals = [float(param_cfg.reference)]

        # CLI overrides (only for the known params):
        if param_name == "inlet_velocity" and args.inlet_velocity is not None:
            # Replace all components with the magnitude override:
            vals = [args.inlet_velocity] * len(vals)
        if param_name == "air_density" and args.air_density is not None:
            vals = [args.air_density]

        # Capture scalar values for physical-unit denormalization:
        if param_name == "inlet_velocity":
            stream_velocity = float(vals[0])
        if param_name == "air_density":
            air_density = float(vals[0])

        gp_vals.extend(vals)
        gp_refs.extend(vals)  # reference = same as value (already normalised in config)

    global_params_values = torch.tensor(
        gp_vals, dtype=torch.float32, device=device
    ).reshape(-1, 1)
    global_params_reference = torch.tensor(
        gp_refs, dtype=torch.float32, device=device
    ).reshape(-1, 1)

    logger.info(f"         global_params: {global_params_values.flatten().tolist()}")
    logger.info(f"         stream_velocity: {stream_velocity} m/s  |  air_density: {air_density} kg/m³")

    # -----------------------------------------------------------------------
    # Load STL
    # -----------------------------------------------------------------------
    _step(logger, 3, N_STEPS, f"Loading STL: {args.stl}")
    stl_coordinates, stl_faces = load_stl_to_tensors(args.stl, device)
    stl_min = stl_coordinates.min(dim=0).values.cpu().tolist()
    stl_max = stl_coordinates.max(dim=0).values.cpu().tolist()
    logger.info(
        f"         {stl_coordinates.shape[0]:,} vertices, "
        f"{stl_faces.shape[0] // 3:,} triangles"
    )
    logger.info(f"         STL bbox min : {[round(v,4) for v in stl_min]}")
    logger.info(f"         STL bbox max : {[round(v,4) for v in stl_max]}")
    cfg_vol_min = list(cfg.data.bounding_box.min)
    cfg_vol_max = list(cfg.data.bounding_box.max)
    cfg_surf_min = list(cfg.data.bounding_box_surface.min)
    cfg_surf_max = list(cfg.data.bounding_box_surface.max)
    logger.info(f"         Config vol   : {cfg_vol_min} → {cfg_vol_max}")
    logger.info(f"         Config surf  : {cfg_surf_min} → {cfg_surf_max}")
    stl_inside_surf = all(
        cfg_surf_min[i] <= stl_min[i] and stl_max[i] <= cfg_surf_max[i]
        for i in range(3)
    )
    stl_inside_vol = all(
        cfg_vol_min[i] <= stl_min[i] and stl_max[i] <= cfg_vol_max[i]
        for i in range(3)
    )
    if not stl_inside_vol:
        logger.warning(
            "STL vertices extend OUTSIDE the config volume bounding box! "
            "Coordinate system mismatch between STL and config."
        )
    elif not stl_inside_surf:
        logger.warning(
            "STL vertices extend outside the config SURFACE bounding box "
            "(this may be expected if the surface bbox is approximate)."
        )
    else:
        logger.info("         STL fits within both config bounding boxes. ✓")

    # -----------------------------------------------------------------------
    # Build datapipe (preprocessing only — no dataset)
    # Grid mode uses sampling=False so every input point is processed as-is.
    # -----------------------------------------------------------------------
    output_ext = Path(args.output).suffix.lower()
    grid_mode = (output_ext == ".vti")

    _step(logger, 4, N_STEPS, "Building preprocessing pipeline …")
    datapipe = build_datapipe(cfg, vol_factors, sampling=False if grid_mode else None)
    logger.info(
        f"         bbox: {list(cfg.data.bounding_box.min)} → "
        f"{list(cfg.data.bounding_box.max)}"
    )
    if grid_mode:
        logger.info("         Mode : GRID — evaluating on VTI cell centres (no random sampling)")

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    _step(logger, 5, N_STEPS, "Building DoMINO model …")
    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(device)

    logger.info(f"         {num_vol_vars} output channels")

    # -----------------------------------------------------------------------
    # Load checkpoint
    # Load the .mdlus model file directly — avoids load_checkpoint also trying
    # to load the training-state .pt file (optimizer/scheduler), which we
    # don't need for inference.
    # -----------------------------------------------------------------------
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    import glob
    import re

    mdlus_files = glob.glob(str(checkpoint_dir / "DoMINO.0.*.mdlus"))
    if not mdlus_files:
        logger.error(f"No DoMINO.0.*.mdlus checkpoint files found in {checkpoint_dir}")
        sys.exit(1)

    # Sort numerically by epoch number to get the latest:
    def _epoch(f):
        m = re.search(r"\.(\d+)\.mdlus$", f)
        return int(m.group(1)) if m else -1

    latest_mdlus = max(mdlus_files, key=_epoch)
    logger.info(f"         Loading: {Path(latest_mdlus).name}")
    model.load(latest_mdlus)
    logger.info(f"         Checkpoint loaded successfully")

    model.eval()

    # -----------------------------------------------------------------------
    # Determine output variable names and canonical rename map
    # -----------------------------------------------------------------------
    channel_names = []
    vti_output_names: dict[str, str] = {}
    _first_vector = True
    _first_scalar = True
    for var_name, var_type in cfg.variables.volume.solution.items():
        if var_type == "vector":
            channel_names += [f"{var_name}_x", f"{var_name}_y", f"{var_name}_z"]
            if _first_vector:
                vti_output_names[var_name] = "velocity_time_avg"
                _first_vector = False
        else:
            channel_names.append(var_name)
            if _first_scalar:
                vti_output_names[var_name] = "pressure_time_avg"
                _first_scalar = False
    logger.info(f"         Output channels: {channel_names}")
    logger.info(f"         VTI rename map:  {vti_output_names}")

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    _step(logger, 6, N_STEPS, "Running inference …")
    batch_size = args.batch_size or cfg.model.volume_points_sample
    t_start = time.perf_counter()

    if grid_mode:
        # --- Grid mode: evaluate at every VTI cell centre ---
        vti_res = (
            tuple(args.vti_resolution)
            if args.vti_resolution is not None
            else tuple(int(r) for r in cfg.model.interp_res)
        )
        nx, ny, nz = vti_res
        logger.info(
            f"         Grid {nx}×{ny}×{nz} = {nx*ny*nz:,} cells, "
            f"batch_size={batch_size:,}"
        )
        preds_flat, bbox_min, bbox_max, spacing = run_inference_grid(
            stl_coordinates=stl_coordinates,
            stl_faces=stl_faces,
            global_params_values=global_params_values,
            global_params_reference=global_params_reference,
            model=model,
            datapipe=datapipe,
            resolution=vti_res,
            batch_size=batch_size,
            logger=logger,
        )
        t_end = time.perf_counter()
        n_exterior = int((preds_flat != 0).any(axis=1).sum())
        logger.info(
            f"Inference complete: {n_exterior:,}/{nx*ny*nz:,} exterior cells "
            f"in {t_end - t_start:.1f}s"
        )
        logger.info(f"Saving {nx}×{ny}×{nz} VTI to: {args.output}")
        save_vti_direct(
            preds_flat, args.output, channel_names,
            bbox_min=bbox_min, bbox_max=bbox_max,
            resolution=vti_res,
            output_names=vti_output_names,
        )
    else:
        # --- Scatter mode: random point sampling → VTU/VTP ---
        n_batches = -(-args.num_points // batch_size)
        logger.info(
            f"         {args.num_points:,} points, "
            f"batch_size={batch_size:,}, ~{n_batches} batches"
        )
        volume_coords, volume_preds = run_inference(
            stl_coordinates=stl_coordinates,
            stl_faces=stl_faces,
            global_params_values=global_params_values,
            global_params_reference=global_params_reference,
            model=model,
            datapipe=datapipe,
            batch_size=batch_size,
            total_points=args.num_points,
            logger=logger,
        )
        t_end = time.perf_counter()
        logger.info(
            f"Inference complete: {volume_coords.shape[0]:,} points "
            f"in {t_end - t_start:.1f}s "
            f"({volume_coords.shape[0] / (t_end - t_start):.0f} pts/s)"
        )
        logger.info(f"Saving {volume_coords.shape[0]:,} points to: {args.output}")
        save_vtu(volume_coords, volume_preds, args.output, channel_names)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info(f"  Output : {args.output}")
    logger.info(f"  Time   : {t_end - t_start:.1f}s")
    if grid_mode:
        logger.info(f"  Grid   : {vti_res[0]}×{vti_res[1]}×{vti_res[2]} — supports all ParaView filters")
    logger.info(f"  Open '{args.output}' in ParaView to visualise results.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
