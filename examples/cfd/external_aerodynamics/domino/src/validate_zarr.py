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
Validate zarr files for DoMINO volume-only training.

Checks every case for:
  - All required keys present
  - Correct shapes and dtypes
  - No NaN or Inf values
  - Self-consistent geometry (faces index within stl_coordinates range)
  - Global params values

Usage:
    python validate_zarr.py --data_dir D:/Downloads/raf_test/zarr_train
    python validate_zarr.py --data_dir D:/Downloads/raf_test/zarr_val
    python validate_zarr.py --data_dir D:/Downloads/raf_test/zarr_train --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

# ── Required keys for volume-only DoMINO ──────────────────────────────────────
REQUIRED_KEYS = [
    "stl_coordinates",      # [N_verts, 3]   float32
    "stl_centers",          # [N_faces, 3]   float32
    "stl_faces",            # [N_faces*3]    int32
    "stl_areas",            # [N_faces]      float32
    "volume_mesh_centers",  # [N_vol, 3]     float32
    "volume_fields",        # [N_vol, C]     float32  — C=4: [Vx,Vy,Vz,P]
    "global_params_values",    # [n_params, 1] float32
    "global_params_reference", # [n_params, 1] float32
]

EXPECTED_VOL_CHANNELS = 4   # Vx, Vy, Vz, P
EXPECTED_GLOBAL_PARAMS = 2  # inlet_velocity, air_density


def validate_case(zarr_path: Path, verbose: bool = False) -> list[str]:
    """
    Validate one zarr case. Returns a list of error strings (empty = pass).
    """
    errors = []
    try:
        store = zarr.open(str(zarr_path), mode="r")
    except Exception as e:
        return [f"Cannot open zarr: {e}"]

    present_keys = list(store.keys())

    # ── 1. Check all required keys are present ─────────────────────────────────
    for key in REQUIRED_KEYS:
        if key not in present_keys:
            errors.append(f"Missing key: '{key}'")

    if errors:  # can't do shape checks if keys are missing
        return errors

    # ── 2. Load arrays ─────────────────────────────────────────────────────────
    stl_coords  = store["stl_coordinates"][:]
    stl_centers = store["stl_centers"][:]
    stl_faces   = store["stl_faces"][:]
    stl_areas   = store["stl_areas"][:]
    vol_centers = store["volume_mesh_centers"][:]
    vol_fields  = store["volume_fields"][:]
    gp_vals     = store["global_params_values"][:]
    gp_ref      = store["global_params_reference"][:]

    n_verts = stl_coords.shape[0]
    n_faces = stl_centers.shape[0]
    n_vol   = vol_centers.shape[0]

    # ── 3. Shape checks ────────────────────────────────────────────────────────
    if stl_coords.ndim != 2 or stl_coords.shape[1] != 3:
        errors.append(f"stl_coordinates shape {stl_coords.shape} expected [N,3]")

    if stl_centers.ndim != 2 or stl_centers.shape[1] != 3:
        errors.append(f"stl_centers shape {stl_centers.shape} expected [N,3]")

    if stl_faces.ndim != 1 or stl_faces.shape[0] != n_faces * 3:
        errors.append(
            f"stl_faces shape {stl_faces.shape} expected [{n_faces*3}] (n_faces*3)"
        )

    if stl_areas.ndim != 1 or stl_areas.shape[0] != n_faces:
        errors.append(f"stl_areas shape {stl_areas.shape} expected [{n_faces}]")

    if vol_centers.ndim != 2 or vol_centers.shape[1] != 3:
        errors.append(f"volume_mesh_centers shape {vol_centers.shape} expected [N,3]")

    if vol_fields.ndim != 2:
        errors.append(f"volume_fields shape {vol_fields.shape} expected [N, C]")
    elif vol_fields.shape[1] != EXPECTED_VOL_CHANNELS:
        errors.append(
            f"volume_fields has {vol_fields.shape[1]} channels, "
            f"expected {EXPECTED_VOL_CHANNELS} (Vx,Vy,Vz,P)"
        )

    if vol_fields.shape[0] != n_vol:
        errors.append(
            f"volume_fields rows {vol_fields.shape[0]} != "
            f"volume_mesh_centers rows {n_vol}"
        )

    if gp_vals.shape != (EXPECTED_GLOBAL_PARAMS, 1):
        errors.append(
            f"global_params_values shape {gp_vals.shape} "
            f"expected ({EXPECTED_GLOBAL_PARAMS}, 1)"
        )

    if gp_ref.shape != (EXPECTED_GLOBAL_PARAMS, 1):
        errors.append(
            f"global_params_reference shape {gp_ref.shape} "
            f"expected ({EXPECTED_GLOBAL_PARAMS}, 1)"
        )

    # ── 4. dtype checks ────────────────────────────────────────────────────────
    for name, arr, expected in [
        ("stl_coordinates",  stl_coords,  np.float32),
        ("stl_centers",      stl_centers, np.float32),
        ("stl_faces",        stl_faces,   np.int32),
        ("stl_areas",        stl_areas,   np.float32),
        ("volume_mesh_centers", vol_centers, np.float32),
        ("volume_fields",    vol_fields,  np.float32),
    ]:
        if arr.dtype != expected:
            errors.append(f"{name} dtype={arr.dtype}, expected {expected}")

    # ── 5. NaN / Inf checks ────────────────────────────────────────────────────
    for name, arr in [
        ("stl_coordinates",     stl_coords),
        ("stl_centers",         stl_centers),
        ("stl_areas",           stl_areas),
        ("volume_mesh_centers", vol_centers),
        ("volume_fields",       vol_fields),
    ]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0:
            errors.append(f"{name} contains {n_nan} NaN values")
        if n_inf > 0:
            errors.append(f"{name} contains {n_inf} Inf values")

    # ── 6. Geometry consistency check ──────────────────────────────────────────
    if stl_faces.size > 0:
        face_max = stl_faces.max()
        face_min = stl_faces.min()
        if face_max >= n_verts:
            errors.append(
                f"stl_faces max index {face_max} >= n_verts {n_verts} (out of range)"
            )
        if face_min < 0:
            errors.append(f"stl_faces contains negative index {face_min}")

    # ── 7. Sanity checks on values ─────────────────────────────────────────────
    if stl_areas.size > 0 and stl_areas.min() <= 0:
        n_nonpos = (stl_areas <= 0).sum()
        errors.append(f"stl_areas has {n_nonpos} non-positive area values")

    # ── Verbose summary ────────────────────────────────────────────────────────
    if verbose and not errors:
        print(f"    stl: {n_verts:,} verts | {n_faces:,} faces")
        print(f"    vol: {n_vol:,} pts | {EXPECTED_VOL_CHANNELS} channels")
        vel_mag = np.linalg.norm(vol_fields[:, :3], axis=1)
        print(f"    velocity magnitude: {vel_mag.min():.2f} – {vel_mag.max():.2f} m/s")
        print(f"    pressure:           {vol_fields[:,3].min():.1f} – {vol_fields[:,3].max():.1f} Pa")
        print(f"    global_params:      {gp_vals.flatten()}")

    return errors


def validate_directory(data_dir: Path, verbose: bool = False) -> bool:
    """Validate all zarr cases in a directory. Returns True if all pass."""
    zarr_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")],
        key=lambda d: d.name,
    )

    if not zarr_dirs:
        print(f"ERROR: No .zarr directories found in {data_dir}")
        return False

    print(f"Validating {len(zarr_dirs)} cases in {data_dir}\n")

    n_pass = 0
    n_fail = 0
    all_pass = True

    for zarr_path in zarr_dirs:
        errors = validate_case(zarr_path, verbose=verbose)
        if errors:
            print(f"  FAIL  {zarr_path.name}")
            for e in errors:
                print(f"        ! {e}")
            n_fail += 1
            all_pass = False
        else:
            status = "PASS"
            print(f"  {status}  {zarr_path.name}")
            if verbose:
                pass  # already printed inside validate_case
            n_pass += 1

    print(f"\nResult: {n_pass} passed, {n_fail} failed")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate zarr files for DoMINO volume-only training."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .zarr case folders",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-case statistics (velocity range, pressure range, etc.)",
    )
    args = parser.parse_args()

    ok = validate_directory(Path(args.data_dir), verbose=args.verbose)
    sys.exit(0 if ok else 1)
