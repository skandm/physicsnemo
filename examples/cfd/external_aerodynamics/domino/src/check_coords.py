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
Coordinate sanity-check: compare the spatial extents of

  1. One or more zarr training cases (stl_coordinates + volume_mesh_centers)
  2. The config bounding boxes (volume + surface)
  3. An optional inference STL

Usage:
    python check_coords.py --zarr_dir /path/to/zarr_train \\
                           --config   conf/config.yaml \\
                           --stl      /path/to/mesh.stl   # optional

All extents are printed side-by-side so mismatches are immediately visible.
"""

import argparse
from pathlib import Path

import numpy as np


def stl_bbox(stl_path: str):
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    mesh = pv.read(stl_path)
    pts = np.array(mesh.points)
    return pts.min(axis=0), pts.max(axis=0)


def zarr_bbox(zarr_path: Path):
    import zarr
    z = zarr.open(str(zarr_path), mode="r")
    stl_v = z["stl_coordinates"][:]
    vol_v = z["volume_mesh_centers"][:]
    return (
        stl_v.min(axis=0), stl_v.max(axis=0),
        vol_v.min(axis=0), vol_v.max(axis=0),
    )


def _row(label, lo, hi):
    lo_s = "[" + ", ".join(f"{v:8.4f}" for v in lo) + "]"
    hi_s = "[" + ", ".join(f"{v:8.4f}" for v in hi) + "]"
    span = hi - lo
    sp_s = "[" + ", ".join(f"{v:7.4f}" for v in span) + "]"
    print(f"  {label:<35s}  {lo_s}  →  {hi_s}   span={sp_s}")


def main():
    p = argparse.ArgumentParser(description="Check coordinate alignment across STL, zarr, and config.")
    p.add_argument("--zarr_dir", required=True, help="Directory of .zarr training cases")
    p.add_argument("--config",   required=True, help="Path to config.yaml")
    p.add_argument("--stl",      default=None,  help="Optional inference STL to check")
    p.add_argument("--n_cases",  type=int, default=3, help="How many zarr cases to sample (default 3)")
    args = p.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    print("\n" + "=" * 90)
    print("COORDINATE EXTENT COMPARISON")
    print("=" * 90)

    # 1. Config bounding boxes
    vol_min  = np.array(cfg.data.bounding_box.min,         dtype=np.float32)
    vol_max  = np.array(cfg.data.bounding_box.max,         dtype=np.float32)
    surf_min = np.array(cfg.data.bounding_box_surface.min, dtype=np.float32)
    surf_max = np.array(cfg.data.bounding_box_surface.max, dtype=np.float32)

    print("\n--- Config bounding boxes ---")
    _row("config  volume bbox",  vol_min,  vol_max)
    _row("config surface bbox", surf_min, surf_max)

    # 2. Zarr training data
    zarr_dir = Path(args.zarr_dir)
    cases = sorted(zarr_dir.glob("*.zarr"))[: args.n_cases]
    if not cases:
        print(f"\nWARNING: no .zarr files found in {zarr_dir}")
    else:
        print(f"\n--- Zarr training cases ({len(cases)} sampled) ---")
        agg_stl_min  =  np.full(3,  np.inf)
        agg_stl_max  =  np.full(3, -np.inf)
        agg_vol_min  =  np.full(3,  np.inf)
        agg_vol_max  =  np.full(3, -np.inf)

        for case in cases:
            try:
                sm, sM, vm, vM = zarr_bbox(case)
                _row(f"  {case.name}  stl",  sm, sM)
                _row(f"  {case.name}  vol",  vm, vM)
                agg_stl_min = np.minimum(agg_stl_min, sm)
                agg_stl_max = np.maximum(agg_stl_max, sM)
                agg_vol_min = np.minimum(agg_vol_min, vm)
                agg_vol_max = np.maximum(agg_vol_max, vM)
            except Exception as e:
                print(f"  ERROR reading {case.name}: {e}")

        if len(cases) > 1:
            print()
            _row("  UNION  stl", agg_stl_min, agg_stl_max)
            _row("  UNION  vol", agg_vol_min, agg_vol_max)

    # 3. Inference STL
    if args.stl:
        print(f"\n--- Inference STL: {args.stl} ---")
        try:
            ism, isM = stl_bbox(args.stl)
            _row("inference STL", ism, isM)
        except Exception as e:
            print(f"  ERROR: {e}")

    # 4. Mismatch warnings
    print("\n--- Checks ---")
    ok = True

    # Does the training STL fit inside the surface bbox?
    if len(cases) > 0:
        if not (np.all(agg_stl_min >= surf_min) and np.all(agg_stl_max <= surf_max)):
            print("  WARNING: zarr stl_coordinates extend OUTSIDE config surface bbox")
            ok = False
        else:
            print("  OK  zarr stl_coordinates fit inside config surface bbox")

        if not (np.all(agg_vol_min >= vol_min) and np.all(agg_vol_max <= vol_max)):
            print("  WARNING: zarr volume_mesh_centers extend OUTSIDE config volume bbox")
            ok = False
        else:
            print("  OK  zarr volume_mesh_centers fit inside config volume bbox")

    if args.stl:
        ism, isM = stl_bbox(args.stl)
        if not (np.all(ism >= surf_min) and np.all(isM <= surf_max)):
            print("  WARNING: inference STL extends OUTSIDE config surface bbox")
            print("           → coordinate system mismatch between inference STL and training data!")
            ok = False
        else:
            print("  OK  inference STL fits inside config surface bbox")

    if ok:
        print("  All checks passed.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
