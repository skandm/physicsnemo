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
Converts STL + CSV files to Zarr format, storing shear (viscous) force instead
of total force as the surface vector field.

Decomposition
-------------
LBM cut-cell total force = pressure contribution + viscous (shear) contribution:

    force_total = -pressure * normal * area  +  shear

Rearranging:

    shear = force_total + pressure * normal * area

The model then learns [pressure, shear_x, shear_y, shear_z].
At inference, total force is reconstructed as:

    total_force = -pred_pressure * normals * areas + pred_shear

Motivation
----------
Total force_x is hard to learn because it is dominated by a large negative spike
at the leading edge (stagnation pressure) with partial positive cancellation at
the trailing edge (pressure recovery).  The net drag is a small residual of two
large opposing terms — a difficult regression target.

Shear force avoids this cancellation: it is distributed smoothly across the
surface, directly correlated with local geometry, and does not have large
leading/trailing edge spikes.  Whether this holds in practice is verified by
the diagnostic statistics printed during conversion.

Expected input layout
---------------------
    input_dir/
        0/
            mesh.stl
            pressure.csv    # columns: x, y, z, p
            force.csv       # columns: x, y, z, fx, fy, fz
        1/
            ...

Output
------
    output_dir/
        0.zarr/   surface_fields = [pressure, shear_x, shear_y, shear_z]
        1.zarr/
        ...

Usage
-----
    1. Set INPUT_DIR, OUTPUT_DIR, and the constants below.
    2. Run once — diagnostic statistics are printed for every case so you can
       verify that shear std < force std before trusting the decomposition.
    3. If shear std ≈ force std or shear std > force std the pressure-contribution
       term is noisy (voxel/STL mismatch) and this approach may not help.
"""

import sys
import numpy as np
import pyvista as pv
import zarr
from pathlib import Path

# ── Input / output directories ────────────────────────────────────────────────
INPUT_DIR  = Path("/data/skand/rafl/batches")
OUTPUT_DIR = Path("/data/skand/rafl/data_nemo_shear")   # separate from force dir

# ── File names inside each case folder ────────────────────────────────────────
STL_FILENAME          = "mesh.stl"
PRESSURE_CSV_FILENAME = "pressure.csv"
FORCE_CSV_FILENAME    = "force.csv"

# ── Subfolder inside each job folder that contains the files ──────────────────
# Set to "" (empty string) if the files are directly in the job folder.
DATA_SUBFOLDER = "outputs"

# ── Column indices in each CSV (0-based) ─────────────────────────────────────
COORD_COLS    = [0, 1, 2]
PRESSURE_COLS = [3]
FORCE_COLS    = [3, 4, 5]

# Set to True if CSVs have a header row, False if they are purely numeric
CSV_HAS_HEADER = False

# ── STL unit conversion ───────────────────────────────────────────────────────
STL_UNIT_SCALE = 0.001   # mm → meters

# ── Global parameters (constant across all cases) ─────────────────────────────
INLET_VELOCITY = 170.0   # m/s
AIR_DENSITY    = 1.225   # kg/m3

# ── Skip already-converted cases (safe to re-run) ─────────────────────────────
SKIP_EXISTING = True


# ──────────────────────────────────────────────────────────────────────────────
# CSV inspection (runs once before the conversion loop)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_csv(input_dir: Path):
    """Print CSV shape info from the first available case."""
    case_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not case_dirs:
        print(f"ERROR: No subdirectories found in {input_dir}")
        sys.exit(1)

    first = case_dirs[0]
    data_dir = first / DATA_SUBFOLDER if DATA_SUBFOLDER else first
    p_csv = data_dir / PRESSURE_CSV_FILENAME
    f_csv = data_dir / FORCE_CSV_FILENAME

    header = 0 if CSV_HAS_HEADER else None

    print(f"\n── CSV inspection (case: {first.name}, subfolder: {DATA_SUBFOLDER or '.'}) ──")
    for label, path in [("pressure.csv", p_csv), ("force.csv", f_csv)]:
        if not path.exists():
            print(f"  {label}: NOT FOUND at {path}")
            continue
        try:
            import pandas as pd
            df = pd.read_csv(path, header=header, nrows=3)
            total_rows = sum(1 for _ in open(path)) - (1 if CSV_HAS_HEADER else 0)
            print(f"  {label}: {total_rows} rows, {df.shape[1]} columns")
            print(f"    first 3 rows:\n{df.to_string(index=False)}")
        except Exception as e:
            print(f"  {label}: could not read ({e})")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (identical to csv_to_zarr.py — kept here so this script is standalone)
# ──────────────────────────────────────────────────────────────────────────────

def read_stl(stl_path: Path, unit_scale: float = 1.0):
    """Read STL and return (data_dict, triangulated_pv_mesh)."""
    stl = pv.read(str(stl_path))
    stl = stl.triangulate()

    faces  = stl.faces.reshape(-1, 4)[:, 1:]
    sizes  = stl.compute_cell_sizes(length=False, area=True, volume=False)
    areas  = np.array(sizes.cell_data["Area"], dtype=np.float32)

    coords  = np.array(stl.points,              dtype=np.float32) * unit_scale
    centers = np.array(stl.cell_centers().points, dtype=np.float32) * unit_scale
    areas   = areas * (unit_scale ** 2)

    stl_scaled = stl.copy()
    stl_scaled.points = coords
    stl_scaled = stl_scaled.compute_normals(cell_normals=True, point_normals=False)

    data_dict = {
        "stl_coordinates": coords,
        "stl_centers":     centers,
        "stl_faces":       faces.flatten().astype(np.int32),
        "stl_areas":       areas,
    }
    return data_dict, stl_scaled


def read_csv_surface(
    pressure_csv: Path,
    force_csv: Path,
    coord_cols: list,
    pressure_cols: list,
    force_cols: list,
    has_header: bool,
    unit_scale: float,
) -> dict:
    """Load cut-cell CSVs and return surface mesh centers and raw [p, fx, fy, fz]."""
    skiprows = 1 if has_header else 0

    p_arr = np.loadtxt(str(pressure_csv), delimiter=",", skiprows=skiprows)
    f_arr = np.loadtxt(str(force_csv),    delimiter=",", skiprows=skiprows)

    p_coords = p_arr[:, coord_cols].astype(np.float32)
    f_coords = f_arr[:, coord_cols].astype(np.float32)

    if p_coords.shape[0] != f_coords.shape[0]:
        raise ValueError(
            f"Row count mismatch: pressure.csv has {p_coords.shape[0]} rows, "
            f"force.csv has {f_coords.shape[0]} rows"
        )
    max_diff = np.abs(p_coords - f_coords).max()
    if max_diff > 1e-6:
        raise ValueError(
            f"Coordinate mismatch between pressure.csv and force.csv "
            f"(max diff = {max_diff:.2e})."
        )

    coords   = p_coords * unit_scale
    pressure = p_arr[:, pressure_cols].astype(np.float32)   # [N, 1]
    force    = f_arr[:, force_cols].astype(np.float32)      # [N, 3]
    fields   = np.concatenate([pressure, force], axis=1)    # [N, 4]: [p, fx, fy, fz]

    valid_mask = np.isfinite(fields).all(axis=1) & np.isfinite(coords).all(axis=1)
    n_invalid  = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"    Dropping {n_invalid} invalid (NaN/Inf) rows")
        fields = fields[valid_mask]
        coords = coords[valid_mask]

    return {
        "surface_mesh_centers": coords,   # [N, 3]
        "surface_fields":       fields,   # [N, 4]: [p, fx, fy, fz]
    }


def compute_surface_features(stl, stl_areas: np.ndarray, surface_pts: np.ndarray):
    """For each cut-cell center, inherit the normal and area of the nearest STL face."""
    cell_ids, _ = stl.find_closest_cell(surface_pts, return_closest_point=True)
    normals = stl.cell_data["Normals"][cell_ids].astype(np.float32)  # [N, 3]
    areas   = stl_areas[cell_ids].astype(np.float32)                 # [N]
    return normals, areas


# ──────────────────────────────────────────────────────────────────────────────
# Shear decomposition
# ──────────────────────────────────────────────────────────────────────────────

def compute_shear(
    surface_fields: np.ndarray,
    surface_normals: np.ndarray,
    surface_areas: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Decompose total force into pressure contribution + shear (viscous).

    Total force on a surface element (continuum):
        F = -p * n̂ * area   (pressure, acts inward against outward normal)
          + shear            (viscous tangential stress)

    Therefore:
        shear = F_total - (-p * n̂ * area)
              = F_total + p * n̂ * area

    Args:
        surface_fields:  [N, 4] array — [pressure, fx, fy, fz]
        surface_normals: [N, 3] outward face normals
        surface_areas:   [N]   face areas

    Returns:
        shear_fields: [N, 4] array — [pressure, shear_x, shear_y, shear_z]
        stats:        dict with std comparison for diagnostic printing
    """
    pressure = surface_fields[:, 0:1]   # [N, 1]
    force    = surface_fields[:, 1:4]   # [N, 3]: [fx, fy, fz]

    # Pressure contribution: F_pressure = -p * n̂ * area  →  shape [N, 3]
    pressure_contrib = -pressure * surface_normals * surface_areas[:, None]

    # Shear = total - pressure contribution
    shear = force - pressure_contrib    # [N, 3]: [shear_x, shear_y, shear_z]

    shear_fields = np.concatenate([pressure, shear], axis=1)  # [N, 4]

    stats = {
        "force_x_std":  float(force[:, 0].std()),
        "shear_x_std":  float(shear[:, 0].std()),
        "force_z_std":  float(force[:, 2].std()),
        "shear_z_std":  float(shear[:, 2].std()),
        "pressure_contrib_x_std": float(pressure_contrib[:, 0].std()),
    }
    return shear_fields, stats


# ──────────────────────────────────────────────────────────────────────────────
# Per-case conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_case(
    case_dir: Path,
    output_dir: Path,
    inlet_velocity: float,
    air_density: float,
    skip_existing: bool,
) -> bool:
    """Convert one case folder to a zarr file with shear surface fields."""
    out_path = output_dir / f"{case_dir.name}.zarr"

    if skip_existing and out_path.exists():
        print(f"  SKIP (already exists): {out_path.name}")
        return True

    data_dir = case_dir / DATA_SUBFOLDER if DATA_SUBFOLDER else case_dir

    stl_path = data_dir / STL_FILENAME
    p_csv    = data_dir / PRESSURE_CSV_FILENAME
    f_csv    = data_dir / FORCE_CSV_FILENAME

    for path in [stl_path, p_csv, f_csv]:
        if not path.exists():
            print(f"  ERROR: {path} not found — skipping")
            return False

    try:
        stl_data, stl_mesh = read_stl(stl_path, unit_scale=STL_UNIT_SCALE)
        csv_data = read_csv_surface(
            pressure_csv  = p_csv,
            force_csv     = f_csv,
            coord_cols    = COORD_COLS,
            pressure_cols = PRESSURE_COLS,
            force_cols    = FORCE_COLS,
            has_header    = CSV_HAS_HEADER,
            unit_scale    = STL_UNIT_SCALE,
        )
    except Exception as e:
        print(f"  ERROR reading {case_dir.name}: {e}")
        return False

    try:
        surface_normals, surface_areas = compute_surface_features(
            stl         = stl_mesh,
            stl_areas   = stl_data["stl_areas"],
            surface_pts = csv_data["surface_mesh_centers"],
        )
    except Exception as e:
        print(f"  ERROR computing surface features for {case_dir.name}: {e}")
        return False

    try:
        shear_fields, stats = compute_shear(
            surface_fields  = csv_data["surface_fields"],
            surface_normals = surface_normals,
            surface_areas   = surface_areas,
        )
    except Exception as e:
        print(f"  ERROR computing shear for {case_dir.name}: {e}")
        return False

    global_params = np.array([[inlet_velocity], [air_density]], dtype=np.float32)

    store = zarr.open(str(out_path), mode="a")

    store["stl_coordinates"] = stl_data["stl_coordinates"]
    store["stl_centers"]     = stl_data["stl_centers"]
    store["stl_faces"]       = stl_data["stl_faces"]
    store["stl_areas"]       = stl_data["stl_areas"]

    store["surface_mesh_centers"] = csv_data["surface_mesh_centers"]
    store["surface_normals"]      = surface_normals
    store["surface_areas"]        = surface_areas
    store["surface_fields"]       = shear_fields   # [p, shear_x, shear_y, shear_z]

    store["global_params_values"]    = global_params
    store["global_params_reference"] = global_params

    n_surf = csv_data["surface_mesh_centers"].shape[0]
    ratio_x = stats["shear_x_std"] / (stats["force_x_std"] + 1e-12)
    ratio_z = stats["shear_z_std"] / (stats["force_z_std"] + 1e-12)
    flag_x  = "  ← WARN: shear noisier than force!" if ratio_x > 0.9 else ""
    flag_z  = "  ← WARN: shear noisier than force!" if ratio_z > 0.9 else ""

    print(
        f"  OK: {n_surf} pts | "
        f"fx std {stats['force_x_std']:.4e} → shear_x std {stats['shear_x_std']:.4e} "
        f"(ratio {ratio_x:.2f}){flag_x} | "
        f"fz std {stats['force_z_std']:.4e} → shear_z std {stats['shear_z_std']:.4e} "
        f"(ratio {ratio_z:.2f}){flag_z}"
    )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not INPUT_DIR.exists():
        print(f"ERROR: INPUT_DIR does not exist: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    print(f"Found {len(case_dirs)} cases in {INPUT_DIR}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"STL unit scale: {STL_UNIT_SCALE}")
    print(f"CSV has header: {CSV_HAS_HEADER}")
    print()
    print("Diagnostic: shear_std / force_std < 1.0 means decomposition reduced variance.")
    print("            ratio > 0.9 → decomposition may be amplifying noise — check!")
    print()

    inspect_csv(INPUT_DIR)

    n_ok   = 0
    n_fail = 0
    for case_dir in case_dirs:
        print(f"[{case_dir.name}]")
        success = convert_case(
            case_dir       = case_dir,
            output_dir     = OUTPUT_DIR,
            inlet_velocity = INLET_VELOCITY,
            air_density    = AIR_DENSITY,
            skip_existing  = SKIP_EXISTING,
        )
        if success:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone: {n_ok} converted, {n_fail} failed")
    if n_fail > 0:
        print("Check errors above for failed cases.")
    print()
    print("If most cases show ratio_x < 0.5 the shear decomposition is clean.")
    print("If ratio_x > 0.9 the pressure-contribution term is too noisy to help.")
