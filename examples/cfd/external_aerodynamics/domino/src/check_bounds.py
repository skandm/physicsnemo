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
Report the aggregate bounding boxes of STL geometry and volume mesh across
all .zarr cases in a directory.

Useful for setting bounding_box and bounding_box_surface in config.yaml:

    data:
      bounding_box:          # from volume_mesh_centers
        min: [x_min, y_min, z_min]
        max: [x_max, y_max, z_max]
      bounding_box_surface:  # from stl_coordinates
        min: [x_min, y_min, z_min]
        max: [x_max, y_max, z_max]

Usage:
    python check_bounds.py --data_dir /data/my_dataset/zarr

    # Limit to a few cases for a quick estimate:
    python check_bounds.py --data_dir /data/zarr --max_cases 5
"""

import argparse
from pathlib import Path

import numpy as np
import zarr


def check_bounds(data_dir: Path, max_cases: int = None) -> None:
    """
    Compute and print aggregate bounding boxes across all zarr cases.

    Args:
        data_dir:  Directory containing <case_id>.zarr folders.
        max_cases: If set, only process this many cases.
    """
    cases = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")],
        key=lambda d: d.name,
    )

    if not cases:
        print(f"ERROR: No .zarr directories found in {data_dir}")
        return

    if max_cases is not None:
        cases = cases[:max_cases]
        print(f"Checking {len(cases)} cases (limited by --max_cases {max_cases})\n")
    else:
        print(f"Checking {len(cases)} cases\n")

    # Accumulators for aggregate bounds
    stl_mins = []
    stl_maxs = []
    vol_mins = []
    vol_maxs = []

    for case in cases:
        try:
            z = zarr.open(str(case), mode="r")
        except Exception as e:
            print(f"  WARNING: Cannot open {case.name}: {e}")
            continue

        has_stl = "stl_coordinates" in z
        has_vol = "volume_mesh_centers" in z

        stl_info = ""
        if has_stl:
            c = z["stl_coordinates"][:]
            stl_mins.append(c.min(axis=0))
            stl_maxs.append(c.max(axis=0))
            n_verts = c.shape[0]
            n_faces = len(z["stl_faces"][:]) // 3 if "stl_faces" in z else "?"
            stl_info = f"stl: {n_verts:,} verts / {n_faces} faces"

        vol_info = ""
        if has_vol:
            v = z["volume_mesh_centers"][:]
            vol_mins.append(v.min(axis=0))
            vol_maxs.append(v.max(axis=0))
            vol_info = f"vol: {v.shape[0]:,} pts"

        print(f"  {case.name:20s}  {stl_info}  {vol_info}")

    def _fmt_bounds(mins, maxs, label):
        if not mins:
            print(f"\n{label}: no data found")
            return
        global_min = np.array(mins).min(axis=0)
        global_max = np.array(maxs).max(axis=0)
        print(f"\n{label} (aggregate across {len(mins)} cases):")
        print(f"  x: {global_min[0]:.4f}  to  {global_max[0]:.4f}")
        print(f"  y: {global_min[1]:.4f}  to  {global_max[1]:.4f}")
        print(f"  z: {global_min[2]:.4f}  to  {global_max[2]:.4f}")
        print(f"\n  config.yaml snippet:")
        print(f"    min: [{global_min[0]:.2f}, {global_min[1]:.2f}, {global_min[2]:.2f}]")
        print(f"    max: [{global_max[0]:.2f}, {global_max[1]:.2f}, {global_max[2]:.2f}]")

    _fmt_bounds(stl_mins, stl_maxs, "STL surface bounds  (bounding_box_surface)")
    _fmt_bounds(vol_mins, vol_maxs, "Volume mesh bounds  (bounding_box)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report bounding boxes of STL and volume mesh across zarr cases."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .zarr case folders",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Only check this many cases (faster for large datasets)",
    )
    args = parser.parse_args()

    check_bounds(Path(args.data_dir), max_cases=args.max_cases)
