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
Surface DoMINO inference — programmatic runner.

Loads a surface-trained (or combined) DoMINO model once and runs inference on
multiple STL geometries, predicting field values (pressure, wall shear, force,
etc.) at every triangle centre of the input mesh.

Output is written as a VTP (ParaView PolyData) file: the original triangulated
surface with predicted fields attached as cell data arrays.

Programmatic usage (multi-geometry):
    from run_surface_inference import SurfaceDoMINORunner

    runner = SurfaceDoMINORunner(
        checkpoint_dir="outputs/LBM_Surface/1/models",
    )
    runner.infer("car_v1.stl", "car_v1.vtp")
    runner.infer("car_v2.stl", "car_v2.vtp", inlet_velocity=40.0)

CLI usage:
    python run_surface_inference.py \\
        --stl        /path/to/mesh.stl \\
        --checkpoint outputs/LBM_Surface/1/models \\
        --output     predicted_surface.vtp \\
        --stl_scale  0.001
"""

# ---------------------------------------------------------------------------
# Environment compatibility shims — must run before any physicsnemo import.
#
# These patches work around version mismatches present on some environments:
#
# Shim 1 — warp.context:
#   warp >= 1.13 removed wp.context; patch it back so downstream code that
#   still references it does not crash on import.
#
# Shim 2 — physicsnemo.models stub:
#   physicsnemo/models/__init__.py eagerly imports all models (DiT, etc.)
#   which pulls in heavy optional deps (timm, torchvision, …) that may not
#   be installed.  We pre-register a lightweight stub with __path__ set so
#   Python can still find subpackages (e.g. physicsnemo.models.domino.model)
#   without executing the top-level __init__.py.
# ---------------------------------------------------------------------------
import os as _os
import sys
import types

import warp as _wp

if not hasattr(_wp, "context"):
    _ctx = types.ModuleType("warp.context")
    _ctx.Device = getattr(_wp, "Device", object)
    _ctx.__getattr__ = lambda name: getattr(_wp, name)  # type: ignore[attr-defined]
    _wp.context = _ctx
    sys.modules["warp.context"] = _ctx

import physicsnemo as _pn  # safe — physicsnemo/__init__.py only imports warp + core

if "physicsnemo.models" not in sys.modules:
    _models_stub = types.ModuleType("physicsnemo.models")
    _models_stub.__path__ = [_os.path.join(_os.path.dirname(_pn.__file__), "models")]
    _models_stub.__package__ = "physicsnemo.models"
    sys.modules["physicsnemo.models"] = _models_stub

# ---------------------------------------------------------------------------
# Standard imports (safe after shims above)
# ---------------------------------------------------------------------------
import argparse
import glob
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.domino.model import DoMINO

from utils import get_num_vars, load_scaling_factors
from run_inference import load_stl_to_tensors
from inference_on_stl import inference_on_single_stl


def _make_logger():
    log = logging.getLogger("run_surface_inference")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        log.addHandler(handler)
    return log


# ---------------------------------------------------------------------------
# Surface datapipe builder (surface_factors != None)
# ---------------------------------------------------------------------------

def build_surface_datapipe(
    cfg,
    vol_factors: Optional[torch.Tensor],
    surf_factors: torch.Tensor,
) -> DoMINODataPipe:
    """Build a DoMINODataPipe configured for surface inference."""
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()

    overrides = {}
    if hasattr(cfg.data, "gpu_preprocessing"):
        overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing
    if hasattr(cfg.data, "gpu_output"):
        overrides["gpu_output"] = cfg.data.gpu_output

    return DoMINODataPipe(
        input_path=None,
        phase="test",
        model_type=cfg.model.model_type,
        grid_resolution=cfg.model.interp_res,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sampling=False,
        sample_in_bbox=False,  # we pre-filter to the surface bbox ourselves
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        volume_sample_from_disk=False,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Multi-geometry runner (load once, infer many)
# ---------------------------------------------------------------------------

class SurfaceDoMINORunner:
    """Load a surface DoMINO model once and run inference on multiple geometries.

    Expensive operations (config load, scaling factors, datapipe construction,
    model weight deserialisation) happen once in ``__init__``.  Each call to
    :meth:`infer` only performs per-geometry work (STL load + model forward).

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory containing ``DoMINO.0.*.mdlus`` checkpoint files.
    config_path : str | Path | None
        Path to ``config.yaml``.  If *None*, auto-detected from
        ``<checkpoint_dir>/../hydra/config.yaml``.
    scaling_path : str | Path | None
        Path to ``scaling_factors.pkl``.  If *None*, read from config.
    device : torch.device | str | None
        Target device.  Defaults to GPU 0 when available.
    """

    def __init__(
        self,
        checkpoint_dir,
        config_path=None,
        scaling_path=None,
        device=None,
    ):
        logger = _make_logger()
        self._logger = logger

        # 1. Device
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        dm = DistributedManager()
        self.device = torch.device(device) if device is not None else dm.device
        logger.info(f"SurfaceDoMINORunner — device: {self.device}")

        # 2. Config
        checkpoint_dir = Path(checkpoint_dir)
        if config_path is not None:
            _cfg_path = Path(config_path).resolve()
        else:
            candidates = [
                checkpoint_dir.parent / "hydra" / "config.yaml",
                checkpoint_dir / "hydra" / "config.yaml",
                Path("conf/config.yaml").resolve(),
            ]
            _cfg_path = next((p for p in candidates if p.exists()), candidates[-1])
            logger.info(f"Auto-detected config: {_cfg_path}")

        if not _cfg_path.exists():
            raise FileNotFoundError(
                f"Config not found: {_cfg_path}\n"
                "Pass config_path= explicitly."
            )

        cfg = OmegaConf.load(_cfg_path)

        if scaling_path is not None:
            OmegaConf.update(cfg, "data.scaling_factors", str(scaling_path), merge=True)

        _scaling_path = Path(cfg.data.scaling_factors)
        if not _scaling_path.exists():
            raise FileNotFoundError(f"Scaling factors not found: {_scaling_path}")

        self.cfg = cfg
        logger.info(f"Config       : {_cfg_path}")
        logger.info(f"Scaling path : {_scaling_path}")

        # 3. Validate model type
        model_type = cfg.model.model_type
        if model_type not in ("surface", "combined"):
            raise ValueError(
                f"Config specifies model_type='{model_type}'. "
                "SurfaceDoMINORunner requires 'surface' or 'combined'."
            )

        # 4. Scaling factors
        logger.info("Loading scaling factors …")
        vol_factors, surf_factors = load_scaling_factors(cfg)
        if surf_factors is None:
            raise ValueError(
                "No 'surface_fields' found in scaling_factors.pkl. "
                "Ensure the file was computed from a surface-trained model."
            )
        logger.info(f"  surf_factors shape: {surf_factors.shape}")

        # 5. Datapipe
        logger.info("Building preprocessing pipeline …")
        self.datapipe = build_surface_datapipe(cfg, vol_factors, surf_factors)

        # 6. Model
        logger.info("Building DoMINO model …")
        num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)
        self.model = DoMINO(
            input_features=3,
            output_features_vol=num_vol_vars,
            output_features_surf=num_surf_vars,
            global_features=num_global_features,
            model_parameters=cfg.model,
        ).to(self.device)

        # 7. Checkpoint
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        mdlus_files = glob.glob(str(checkpoint_dir / "DoMINO.0.*.mdlus"))
        if not mdlus_files:
            mdlus_files = glob.glob(str(checkpoint_dir / "*.mdlus"))
        if not mdlus_files:
            raise FileNotFoundError(f"No .mdlus files found in {checkpoint_dir}")

        def _epoch(f):
            m = re.search(r"\.(\d+)\.mdlus$", f)
            return int(m.group(1)) if m else -1

        latest = max(mdlus_files, key=_epoch)
        logger.info(f"Loading checkpoint: {Path(latest).name}")
        self.model.load(latest)
        self.model.eval()
        logger.info("Checkpoint loaded — model ready.")

        # 8. Output channel names
        self.channel_names: list[str] = []
        for var_name, var_type in cfg.variables.surface.solution.items():
            if var_type == "vector":
                self.channel_names += [f"{var_name}_x", f"{var_name}_y", f"{var_name}_z"]
            else:
                self.channel_names.append(var_name)
        logger.info(f"Surface output channels: {self.channel_names}")

        # 9. Default global parameter tensors
        gp = cfg.variables.global_parameters
        gp_vals, gp_refs = [], []
        for param_cfg in gp.values():
            ref = param_cfg.reference
            refs = list(ref) if hasattr(ref, "__iter__") else [float(ref)]
            gp_vals.extend(refs)
            gp_refs.extend(refs)
        self._default_gp_vals = torch.tensor(gp_vals, dtype=torch.float32, device=self.device)
        self._default_gp_refs = torch.tensor(gp_refs, dtype=torch.float32, device=self.device)

        self._gp_offsets: dict[str, int] = {}
        offset = 0
        for name, param_cfg in gp.items():
            self._gp_offsets[name] = offset
            ref = param_cfg.reference
            offset += len(list(ref)) if hasattr(ref, "__iter__") else 1

    # -------------------------------------------------------------------------

    def infer(
        self,
        stl_path,
        output_dir,
        *,
        inlet_velocity: Optional[float] = None,
        air_density: Optional[float] = None,
        batch_size: Optional[int] = None,
        stl_scale: float = 1.0,
    ) -> str:
        """Run surface inference on a single STL geometry.

        Parameters
        ----------
        stl_path : str | Path
            Input STL file.
        output_dir : str | Path
            Directory where output files are written:
            ``pressure.csv``, ``force.csv``, ``results.json``.
        inlet_velocity : float | None
            Override inlet velocity (m/s).  Uses config default if *None*.
        air_density : float | None
            Override air density (kg/m³).  Uses config default if *None*.
        batch_size : int | None
            Surface points per inference batch.  Defaults to
            ``surface_points_sample`` from config.
        stl_scale : float
            Multiply STL coordinates by this factor before inference
            (e.g. ``0.001`` to convert mm → m).

        Returns
        -------
        str
            Path to ``results.json``.
        """
        logger = self._logger
        cfg = self.cfg
        device = self.device
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"infer: {stl_path} → {output_dir}")

        # Build global params, optionally overriding from caller
        gp_vals = self._default_gp_vals.clone()
        gp_refs = self._default_gp_refs.clone()
        if inlet_velocity is not None and "inlet_velocity" in self._gp_offsets:
            gp_vals[self._gp_offsets["inlet_velocity"]] = float(inlet_velocity)
        if air_density is not None and "air_density" in self._gp_offsets:
            gp_vals[self._gp_offsets["air_density"]] = float(air_density)

        # Load STL
        logger.info(f"  Loading STL: {stl_path}")
        stl_coordinates, stl_faces = load_stl_to_tensors(str(stl_path), device)
        if stl_scale != 1.0:
            stl_coordinates = stl_coordinates * stl_scale
        n_tri_total = stl_faces.shape[0] // 3
        logger.info(
            f"  STL: {stl_coordinates.shape[0]} vertices, {n_tri_total} triangles"
        )

        # -----------------------------------------------------------------------
        # Pre-filter faces to the surface bounding box before calling the model.
        #
        # For surface inference the right domain is bounding_box_surface, not
        # the volume bbox.  We filter here so the datapipe receives only valid
        # triangles (sample_in_bbox is forced False in build_surface_datapipe).
        # -----------------------------------------------------------------------
        faces_2d = stl_faces.reshape(n_tri_total, 3)         # (n_tri, 3)
        tri_verts = stl_coordinates[faces_2d]                 # (n_tri, 3, 3)
        stl_centers_all = tri_verts.mean(dim=1)               # (n_tri, 3)
        d1 = tri_verts[:, 1] - tri_verts[:, 0]
        d2 = tri_verts[:, 2] - tri_verts[:, 0]
        stl_areas_all = 0.5 * torch.linalg.norm(
            torch.linalg.cross(d1, d2, dim=1), dim=1
        )

        valid_mask = stl_areas_all > 0

        if self.datapipe.config.bounding_box_dims_surf is not None:
            s_max = self.datapipe.config.bounding_box_dims_surf[0]
            s_min = self.datapipe.config.bounding_box_dims_surf[1]
            in_surf_bbox = (
                (stl_centers_all > s_min).all(dim=-1)
                & (stl_centers_all < s_max).all(dim=-1)
            )
            valid_mask = valid_mask & in_surf_bbox

        stl_faces_filtered = faces_2d[valid_mask].flatten()
        logger.info(
            f"  Triangles within surface bbox: "
            f"{valid_mask.sum().item()} / {n_tri_total}"
        )

        _batch_size = batch_size or cfg.model.surface_points_sample

        # Compute geometry for the filtered triangles (needed for batched inference)
        n_valid = stl_faces_filtered.shape[0] // 3
        faces_2d = stl_faces_filtered.reshape(n_valid, 3)
        tri_verts = stl_coordinates[faces_2d]            # (n_valid, 3, 3)
        stl_centers = tri_verts.mean(dim=1)              # (n_valid, 3)
        d1 = tri_verts[:, 1] - tri_verts[:, 0]
        d2 = tri_verts[:, 2] - tri_verts[:, 0]
        cross = torch.linalg.cross(d1, d2, dim=1)
        normals_norm = torch.linalg.norm(cross, dim=1)
        stl_normals = cross / normals_norm.unsqueeze(1)  # unit normals
        stl_areas_f = 0.5 * normals_norm

        # Batched inference — process triangle centers in _batch_size chunks to
        # avoid CUDA OOM that occurs when feeding all centers in one forward pass.
        t0 = time.perf_counter()
        chunks = []
        for start in range(0, n_valid, _batch_size):
            end = min(start + _batch_size, n_valid)
            batch_dict = {
                "stl_coordinates": stl_coordinates,
                "stl_faces": stl_faces_filtered,
                "stl_centers": stl_centers[start:end],
                "stl_areas": stl_areas_f[start:end],
                "global_params_values": gp_vals,
                "global_params_reference": gp_refs,
                "surface_mesh_centers": stl_centers[start:end],
                "surface_normals": stl_normals[start:end],
                "surface_areas": stl_areas_f[start:end],
                "surface_faces": stl_faces_filtered,
            }
            preprocessed = self.datapipe.process_data(batch_dict)
            preprocessed = {k: v.unsqueeze(0) for k, v in preprocessed.items()}
            with torch.no_grad():
                _, output_surf = self.model(preprocessed)
            _, chunk_preds = self.datapipe.unscale_model_outputs(None, output_surf)
            chunks.append(chunk_preds.squeeze(0).cpu())
            logger.info(f"  Centers {end}/{n_valid}")

        logger.info(f"  Inference took {time.perf_counter() - t0:.1f}s")
        preds = torch.cat(chunks, dim=0).numpy()        # (n_valid, n_vars)
        centers_np = stl_centers.cpu().numpy()          # (n_valid, 3)
        areas_np = stl_areas_f.cpu().numpy()            # (n_valid,)

        results_path = self._save_outputs(
            centers=centers_np,
            areas=areas_np,
            predictions=preds,
            output_dir=output_dir,
        )
        logger.info(f"  Saved → {output_dir}")
        return results_path

    def _save_outputs(
        self,
        centers: np.ndarray,
        areas: np.ndarray,
        predictions: np.ndarray,
        output_dir: Path,
    ) -> str:
        """Write pressure.csv, force.csv, and results.json to output_dir.

        pressure.csv  — x, y, z, pressure  (one row per triangle centre)
        force.csv     — x, y, z, force_x, force_y, force_z
        results.json  — aggregate stats + file paths
        """
        ch = {name: i for i, name in enumerate(self.channel_names)}

        # pressure.csv
        pressure = predictions[:, ch["pressure"]]
        np.savetxt(
            output_dir / "pressure.csv",
            np.column_stack([centers, pressure]),
            delimiter=",",
        )

        # force.csv
        force_cols = np.column_stack([
            centers,
            predictions[:, ch["force_x"]],
            predictions[:, ch["force_y"]],
            predictions[:, ch["force_z"]],
        ])
        np.savetxt(
            output_dir / "force.csv",
            force_cols,
            delimiter=",",
        )

        # results.json — per-field stats + surface-integrated totals
        total_force_x = float((predictions[:, ch["force_x"]] * areas).sum())
        total_force_y = float((predictions[:, ch["force_y"]] * areas).sum())
        total_force_z = float((predictions[:, ch["force_z"]] * areas).sum())

        results = {
            "n_triangles": int(len(pressure)),
            "pressure": {
                "min": float(pressure.min()),
                "max": float(pressure.max()),
                "mean": float(pressure.mean()),
            },
            "integrated_force": {
                "force_x": total_force_x,
                "force_y": total_force_y,
                "force_z": total_force_z,
            },
            "files": {
                "pressure": "pressure.csv",
                "force": "force.csv",
            },
        }
        results_path = output_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2))
        return str(results_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Surface DoMINO inference — predict fields at STL triangle centres."
    )
    p.add_argument("--stl",        required=True,  help="Input STL file")
    p.add_argument("--checkpoint", required=True,  help="Directory with .mdlus checkpoint files")
    p.add_argument("--output_dir", default="output", help="Directory for pressure.csv, force.csv, results.json")
    p.add_argument("--config",     default=None,   help="Path to config.yaml (auto-detected if omitted)")
    p.add_argument("--scaling",    default=None,   help="Path to scaling_factors.pkl (auto-detected if omitted)")
    p.add_argument("--stl_scale",  type=float, default=1.0, help="Coordinate scale (e.g. 0.001 for mm→m)")
    p.add_argument("--inlet_velocity", type=float, default=None, help="Override inlet velocity (m/s)")
    p.add_argument("--air_density",    type=float, default=None, help="Override air density (kg/m³)")
    p.add_argument("--batch_size",     type=int,   default=None, help="Surface points per inference batch")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = SurfaceDoMINORunner(
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        scaling_path=args.scaling,
    )
    runner.infer(
        stl_path=args.stl,
        output_dir=args.output_dir,
        inlet_velocity=args.inlet_velocity,
        air_density=args.air_density,
        batch_size=args.batch_size,
        stl_scale=args.stl_scale,
    )
