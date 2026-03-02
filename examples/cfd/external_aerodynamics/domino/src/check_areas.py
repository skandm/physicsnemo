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
Report STL face area statistics across all .zarr cases in a directory and
suggest an appropriate area_weighing_factor for config.yaml.

The area_weighing_factor is used to balance the surface integral loss in
DoMINO training. A common heuristic is to set it to 1 / max_area so that
the largest face contributes a unit weight.

    model:
      loss_function:
        area_weighing_factor: <suggested value>

Usage:
    python check_areas.py --data_dir /data/my_dataset/zarr_train

    # Limit cases for a quick estimate:
    python check_areas.py --data_dir /data/zarr_train --max_cases 10
"""

import argparse
from pathlib import Path

import numpy as np
import zarr


def check_areas(data_dir: Path, max_cases: int = None) -> None:
    """
    Compute STL face area statistics across all zarr cases.

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

    all_areas = []

    for case in cases:
        try:
            z = zarr.open(str(case), mode="r")
        except Exception as e:
            print(f"  WARNING: Cannot open {case.name}: {e}")
            continue

        if "stl_areas" not in z:
            print(f"  WARNING: {case.name} has no 'stl_areas' key — skipping")
            continue

        areas = z["stl_areas"][:]
        all_areas.append(areas)
        print(
            f"  {case.name:20s}  "
            f"min={areas.min():.3e}  max={areas.max():.3e}  mean={areas.mean():.3e} m²"
        )

    if not all_areas:
        print("No area data found.")
        return

    combined = np.concatenate(all_areas)
    global_max  = combined.max()
    global_mean = combined.mean()

    print(f"\nAcross {len(all_areas)} cases:")
    print(f"  max area:  {global_max:.4e} m²")
    print(f"  mean area: {global_mean:.4e} m²")
    print(f"  min area:  {combined.min():.4e} m²")
    print(f"\n  Suggested area_weighing_factor (1 / max_area): ~{1 / global_max:.0f}")
    print(f"\n  config.yaml snippet:")
    print(f"    model:")
    print(f"      loss_function:")
    print(f"        area_weighing_factor: {1 / global_max:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report STL area stats and suggest area_weighing_factor for config.yaml."
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

    check_areas(Path(args.data_dir), max_cases=args.max_cases)
