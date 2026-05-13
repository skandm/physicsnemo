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
Converts a dataset of STL + CSV files to Zarr format for DoMINO surface training.

Each case has an STL geometry file and two CSV files containing LBM cut-cell data
(voxels intersected by the STL surface). Surface normals and areas are derived by
projecting each cut-cell center onto the nearest STL face.

Expected input layout:
    input_dir/
        0/
            mesh.stl
            pressure.csv    # columns: x, y, z, p
            velocity.csv    # columns: x, y, z, vx, vy, vz
        1/
            mesh.stl
            pressure.csv
            velocity.csv
        ...

Each case produces one output file:
    output_dir/
        0.zarr/
        1.zarr/
        ...

Composable with convert_to_zarr.py:
    - Surface-only:  run this script alone → creates zarr with STL + surface keys
    - Combined:      run convert_to_zarr.py first (volume keys), then this script
                     second → appends surface keys to the same zarr store

Usage:
    1. Set INPUT_DIR, OUTPUT_DIR, and the other constants below.
    2. Run once — the script prints CSV column shapes before converting.
    3. Adjust COORD_COLS / PRESSURE_COLS / VELOCITY_COLS if your columns differ.
"""

import sys
import numpy as np
import pyvista as pv
import zarr
from pathlib import Path

# ── Input / output directories ────────────────────────────────────────────────
INPUT_DIR  = Path("/data/skand/rafl/batches")
OUTPUT_DIR = Path("/data/skand/rafl/data_nemo")

# ── File names inside each case folder ────────────────────────────────────────
STL_FILENAME          = "mesh.stl"
PRESSURE_CSV_FILENAME = "pressure.csv"
VELOCITY_CSV_FILENAME = "velocity.csv"

# ── Subfolder inside each job folder that contains the files ──────────────────
# Set to "" (empty string) if the files are directly in the job folder.
DATA_SUBFOLDER = "outputs"

# ── Column indices in each CSV (0-based) ─────────────────────────────────────
# pressure.csv expected columns: x, y, z, p
# velocity.csv expected columns: x, y, z, vx, vy, vz
COORD_COLS    = [0, 1, 2]
PRESSURE_COLS = [3]
VELOCITY_COLS = [3, 4, 5]

# Set to True if CSVs have a header row, False if they are purely numeric
CSV_HAS_HEADER = False

# ── STL unit conversion ───────────────────────────────────────────────────────
# Set to 0.001 if STL is in mm and CSV coordinates are in meters (or vice versa)
# Set to 1.0 if both are already in the same units
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
    v_csv = data_dir / VELOCITY_CSV_FILENAME

    header = 0 if CSV_HAS_HEADER else None

    print(f"\n── CSV inspection (case: {first.name}, subfolder: {DATA_SUBFOLDER or '.'}) ──")
    for label, path in [("pressure.csv", p_csv), ("velocity.csv", v_csv)]:
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
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_stl(stl_path: Path, unit_scale: float = 1.0):
    """Read STL and return (data_dict, triangulated_pv_mesh).

    Args:
        unit_scale: Multiply all coordinates by this factor.
                    Use 0.001 to convert mm → meters.
                    Areas are scaled by unit_scale**2 automatically.

    Returns:
        data_dict: dict with stl_coordinates, stl_centers, stl_faces, stl_areas
        stl:       triangulated pyvista PolyData (for nearest-face projection)
    """
    stl = pv.read(str(stl_path))
    stl = stl.triangulate()  # ensure all faces are triangles

    faces = stl.faces.reshape(-1, 4)[:, 1:]  # [N_faces, 3]
    sizes = stl.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.array(sizes.cell_data["Area"], dtype=np.float32)

    coords  = np.array(stl.points, dtype=np.float32) * unit_scale
    centers = np.array(stl.cell_centers().points, dtype=np.float32) * unit_scale
    areas   = areas * (unit_scale ** 2)   # area scales as length^2

    # Return a scaled copy of the mesh for projection queries
    stl_scaled = stl.copy()
    stl_scaled.points = coords
    stl_scaled = stl_scaled.compute_normals(cell_normals=True, point_normals=False)

    data_dict = {
        "stl_coordinates": coords,                              # [N_verts, 3]
        "stl_centers":     centers,                            # [N_faces, 3]
        "stl_faces":       faces.flatten().astype(np.int32),  # [N_faces*3]
        "stl_areas":       areas,                              # [N_faces]
    }
    return data_dict, stl_scaled


def read_csv_surface(
    pressure_csv: Path,
    velocity_csv: Path,
    coord_cols: list,
    pressure_cols: list,
    velocity_cols: list,
    has_header: bool,
    unit_scale: float,
) -> dict:
    """Load cut-cell CSVs and return surface mesh centers and stacked fields.

    Args:
        pressure_csv:  Path to pressure CSV (columns: x, y, z, p, ...)
        velocity_csv:  Path to velocity CSV (columns: x, y, z, vx, vy, vz, ...)
        coord_cols:    Column indices for x, y, z coordinates (same in both files)
        pressure_cols: Column indices for pressure values in pressure_csv
        velocity_cols: Column indices for velocity values in velocity_csv
        has_header:    True if CSVs have a header row
        unit_scale:    Multiply coordinates by this factor (e.g. 0.001 for mm→m)

    Returns:
        dict with:
            "surface_mesh_centers": float32 array [N, 3]
            "surface_fields":       float32 array [N, 4]  — [p, vx, vy, vz]
    """
    skiprows = 1 if has_header else 0

    p_arr = np.loadtxt(str(pressure_csv), delimiter=",", skiprows=skiprows)
    v_arr = np.loadtxt(str(velocity_csv), delimiter=",", skiprows=skiprows)

    # Coordinates from pressure CSV; assert they match velocity CSV
    p_coords = p_arr[:, coord_cols].astype(np.float32)
    v_coords = v_arr[:, coord_cols].astype(np.float32)

    if p_coords.shape[0] != v_coords.shape[0]:
        raise ValueError(
            f"Row count mismatch: pressure.csv has {p_coords.shape[0]} rows, "
            f"velocity.csv has {v_coords.shape[0]} rows"
        )
    max_diff = np.abs(p_coords - v_coords).max()
    if max_diff > 1e-6:
        raise ValueError(
            f"Coordinate mismatch between pressure.csv and velocity.csv "
            f"(max diff = {max_diff:.2e}). Check that all files share the same grid."
        )

    coords = p_coords * unit_scale  # [N, 3]

    # Stack fields: [p, vx, vy, vz] → [N, 4]
    pressure = p_arr[:, pressure_cols].astype(np.float32)   # [N, 1]
    velocity = v_arr[:, velocity_cols].astype(np.float32)   # [N, 3]
    surface_fields = np.concatenate([pressure, velocity], axis=1)  # [N, 4]

    # Drop rows with NaN or Inf
    valid_mask = np.isfinite(surface_fields).all(axis=1) & np.isfinite(coords).all(axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"    Dropping {n_invalid} invalid (NaN/Inf) rows")
        surface_fields = surface_fields[valid_mask]
        coords         = coords[valid_mask]

    return {
        "surface_mesh_centers": coords,         # [N, 3]
        "surface_fields":       surface_fields, # [N, 4]
    }


def compute_surface_features(stl, stl_areas: np.ndarray, surface_pts: np.ndarray):
    """For each cut-cell center, inherit the normal and area of the nearest STL face.

    Args:
        stl:         Triangulated pyvista PolyData with cell normals computed
        stl_areas:   Per-face area array [N_faces]
        surface_pts: Cut-cell center coordinates [N, 3]

    Returns:
        normals: float32 array [N, 3] — outward face normals
        areas:   float32 array [N]   — face areas
    """
    cell_ids, _ = stl.find_closest_cell(surface_pts, return_closest_point=True)
    normals = stl.cell_data["Normals"][cell_ids].astype(np.float32)  # [N, 3]
    areas   = stl_areas[cell_ids].astype(np.float32)                 # [N]
    return normals, areas


def convert_case(
    case_dir: Path,
    output_dir: Path,
    inlet_velocity: float,
    air_density: float,
    skip_existing: bool,
) -> bool:
    """Convert one case folder to a zarr file. Returns True on success.

    Opens the zarr store with mode='a' so this script can append surface keys
    to a zarr that was already created by convert_to_zarr.py (combined mode),
    or create a new surface-only zarr if run standalone.
    """
    out_path = output_dir / f"{case_dir.name}.zarr"

    if skip_existing and out_path.exists():
        print(f"  SKIP (already exists): {out_path.name}")
        return True

    # Resolve data subfolder (e.g. job_folder/output/)
    data_dir = case_dir / DATA_SUBFOLDER if DATA_SUBFOLDER else case_dir

    stl_path = data_dir / STL_FILENAME
    p_csv    = data_dir / PRESSURE_CSV_FILENAME
    v_csv    = data_dir / VELOCITY_CSV_FILENAME

    for path in [stl_path, p_csv, v_csv]:
        if not path.exists():
            print(f"  ERROR: {path} not found — skipping")
            return False

    try:
        stl_data, stl_mesh = read_stl(stl_path, unit_scale=STL_UNIT_SCALE)
        csv_data = read_csv_surface(
            pressure_csv  = p_csv,
            velocity_csv  = v_csv,
            coord_cols    = COORD_COLS,
            pressure_cols = PRESSURE_COLS,
            velocity_cols = VELOCITY_COLS,
            has_header    = CSV_HAS_HEADER,
            unit_scale    = STL_UNIT_SCALE,
        )
    except Exception as e:
        print(f"  ERROR reading {case_dir.name}: {e}")
        return False

    try:
        surface_normals, surface_areas = compute_surface_features(
            stl        = stl_mesh,
            stl_areas  = stl_data["stl_areas"],
            surface_pts= csv_data["surface_mesh_centers"],
        )
    except Exception as e:
        print(f"  ERROR computing surface features for {case_dir.name}: {e}")
        return False

    # Global parameters: shape [n_params, 1]
    # Order must match variables.global_parameters in config.yaml
    global_params = np.array([[inlet_velocity], [air_density]], dtype=np.float32)

    # mode="a": create if not exists, append/overwrite arrays if it does exist.
    # This makes csv_to_zarr.py composable with convert_to_zarr.py.
    store = zarr.open(str(out_path), mode="a")

    # STL keys (harmless duplicate write if zarr was already created by convert_to_zarr.py)
    store["stl_coordinates"] = stl_data["stl_coordinates"]
    store["stl_centers"]     = stl_data["stl_centers"]
    store["stl_faces"]       = stl_data["stl_faces"]
    store["stl_areas"]       = stl_data["stl_areas"]

    # Surface keys
    store["surface_mesh_centers"] = csv_data["surface_mesh_centers"]
    store["surface_normals"]      = surface_normals
    store["surface_areas"]        = surface_areas
    store["surface_fields"]       = csv_data["surface_fields"]

    # Global parameter keys
    store["global_params_values"]    = global_params
    store["global_params_reference"] = global_params

    n_verts = stl_data["stl_coordinates"].shape[0]
    n_faces = stl_data["stl_centers"].shape[0]
    n_surf  = csv_data["surface_mesh_centers"].shape[0]
    n_ch    = csv_data["surface_fields"].shape[1]
    print(
        f"  OK: stl={n_verts} verts / {n_faces} faces | "
        f"surface={n_surf} pts / {n_ch} channels [p, vx, vy, vz, fx, fy, fz]"
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
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"STL unit scale: {STL_UNIT_SCALE}")
    print(f"CSV has header: {CSV_HAS_HEADER}")

    # One-time inspection of CSV shapes before starting conversion
    inspect_csv(INPUT_DIR)

    n_ok   = 0
    n_fail = 0
    for case_dir in case_dirs:
        print(f"[{case_dir.name}]")
        success = convert_case(
            case_dir      = case_dir,
            output_dir    = OUTPUT_DIR,
            inlet_velocity = INLET_VELOCITY,
            air_density   = AIR_DENSITY,
            skip_existing = SKIP_EXISTING,
        )
        if success:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone: {n_ok} converted, {n_fail} failed")
    if n_fail > 0:
        print("Check errors above for failed cases")
