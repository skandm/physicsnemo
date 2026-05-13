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
Converts a dataset of STL + VTI files to Zarr format for DoMINO training.

Expected input layout:
    input_dir/
        0/
            mesh.stl
            result.vti
        1/
            mesh.stl
            result.vti
        ...

Each case produces one output file:
    output_dir/
        0.zarr/
        1.zarr/
        ...

Usage:
    1. Set INPUT_DIR, OUTPUT_DIR, and the other constants below.
    2. Leave VOLUME_FIELD_NAMES = {} on the first run.
       The script will print all available fields from your VTI and exit.
    3. Fill in VOLUME_FIELD_NAMES with the names printed in step 2 and run again.
       If a field name is wrong the script will error immediately and show
       the correct names.
"""

import sys
import numpy as np
import pyvista as pv
import zarr
from pathlib import Path

# ── Input / output directories ────────────────────────────────────────────────
INPUT_DIR  = Path("/data/skand/rafl/batches")   # parent folder containing 0/, 1/, 2/ ...
OUTPUT_DIR = Path("/data/skand/rafl/data_domino")   # where .zarr files will be written

# ── File names inside each case folder ────────────────────────────────────────
STL_FILENAME = "mesh.stl"
VTI_FILENAME = "cfdAnalysis.vti"

# ── VTI field names → leave empty {} to auto-discover ────────────────────────
# Map from VTI field name → "vector" or "scalar"
# Order here determines column order in volume_fields — must match config.yaml
# Leave as {} on the first run: the script will print available field names and exit.
VOLUME_FIELD_NAMES = {
    "Velocity time-averaged": "vector",   # columns 0, 1, 2
    "Pressure time-averaged": "scalar",   # column 3
}

# Whether your fields are in cell_data or point_data (check discovery output)
FIELD_LOCATION = "point_data"

# ── STL unit conversion ───────────────────────────────────────────────────────
# Set to 0.001 if STL is in mm and VTI is in meters (your case)
# Set to 1.0 if both are already in the same units
STL_UNIT_SCALE = 0.001   # mm → meters

# ── Global parameters (constant across all cases) ─────────────────────────────
# Change these to match your simulation conditions
INLET_VELOCITY = 170.0   # m/s  — Mach 0.5 at standard sea level (340 m/s * 0.5)
AIR_DENSITY    = 1.225   # kg/m3 — standard sea level density

# ── Skip already-converted cases (safe to re-run) ─────────────────────────────
SKIP_EXISTING = True


# ──────────────────────────────────────────────────────────────────────────────
# Discovery mode: inspect one VTI and exit
# ──────────────────────────────────────────────────────────────────────────────

def discover_fields(input_dir: Path):
    """Print all fields in the first available VTI file and exit."""
    case_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not case_dirs:
        print(f"ERROR: No subdirectories found in {input_dir}")
        sys.exit(1)

    first_case = case_dirs[0]
    vti_path = first_case / VTI_FILENAME
    stl_path = first_case / STL_FILENAME

    if not vti_path.exists():
        print(f"ERROR: {vti_path} not found")
        sys.exit(1)

    print(f"\nReading: {vti_path}\n")
    mesh = pv.read(str(vti_path))

    print(f"Type:       {type(mesh).__name__}")
    print(f"Dimensions: {mesh.dimensions}")
    print(f"N points:   {mesh.n_points}")
    print(f"N cells:    {mesh.n_cells}")

    print(f"\n── cell_data fields ({len(mesh.cell_data)}) ──")
    for name in mesh.cell_data.keys():
        arr = mesh.cell_data[name]
        print(f"  '{name}': shape={arr.shape}, dtype={arr.dtype}")

    print(f"\n── point_data fields ({len(mesh.point_data)}) ──")
    for name in mesh.point_data.keys():
        arr = mesh.point_data[name]
        print(f"  '{name}': shape={arr.shape}, dtype={arr.dtype}")

    if stl_path.exists():
        stl = pv.read(str(stl_path))
        print(f"\n── STL ──")
        print(f"  N vertices: {stl.n_points}")
        print(f"  N faces:    {stl.n_cells}")
        print(f"  Bounds:     {stl.bounds}")

    # Print a ready-to-paste VOLUME_FIELD_NAMES template
    all_fields = {
        **{n: mesh.cell_data[n] for n in mesh.cell_data.keys()},
        **{n: mesh.point_data[n] for n in mesh.point_data.keys()},
    }
    print("\n── Suggested VOLUME_FIELD_NAMES (edit kind: vector/scalar as needed) ──")
    print("VOLUME_FIELD_NAMES = {")
    for name, arr in all_fields.items():
        kind = "vector" if (arr.ndim == 2 and arr.shape[1] == 3) else "scalar"
        print(f'    "{name}": "{kind}",')
    print("}")
    print(
        "\nFill in VOLUME_FIELD_NAMES above with the fields you want, "
        "then run again to convert all cases."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_stl(stl_path: Path, unit_scale: float = 1.0) -> dict:
    """Read STL and return arrays needed by DoMINO.

    Args:
        unit_scale: Multiply all coordinates by this factor.
                    Use 0.001 to convert mm → meters.
                    Areas are scaled by unit_scale**2 automatically.
    """
    stl = pv.read(str(stl_path))
    stl = stl.triangulate()  # ensure all faces are triangles

    faces = stl.faces.reshape(-1, 4)[:, 1:]  # [N_faces, 3]
    sizes = stl.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.array(sizes.cell_data["Area"], dtype=np.float32)

    coords  = np.array(stl.points, dtype=np.float32) * unit_scale
    centers = np.array(stl.cell_centers().points, dtype=np.float32) * unit_scale
    areas   = areas * (unit_scale ** 2)   # area scales as length^2

    return {
        "stl_coordinates": coords,                          # [N_verts, 3]
        "stl_centers":     centers,                        # [N_faces, 3]
        "stl_faces":       faces.flatten().astype(np.int32), # [N_faces*3]
        "stl_areas":       areas,                          # [N_faces]
    }


def read_vti(vti_path: Path, field_names: dict, field_location: str) -> dict:
    """Read VTI and return volume mesh centers and stacked field array."""
    mesh = pv.read(str(vti_path))

    field_src = (
        mesh.cell_data if field_location == "cell_data" else mesh.point_data
    )

    # Validate all requested fields exist
    missing = [name for name in field_names if name not in field_src]
    if missing:
        raise KeyError(
            f"{vti_path}: fields {missing} not found in {field_location}. "
            f"Available: {list(field_src.keys())}"
        )

    # Get coordinates of field evaluation points
    if field_location == "cell_data":
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)
    else:
        coords = np.array(mesh.points, dtype=np.float32)

    # Stack fields into [N, total_channels] in the declared order
    columns = []
    for name, kind in field_names.items():
        arr = np.array(field_src[name], dtype=np.float32)
        if kind == "vector":
            if arr.ndim == 1 or arr.shape[1] != 3:
                raise ValueError(
                    f"Field '{name}' declared as vector but has shape {arr.shape}"
                )
            columns.append(arr)        # [N, 3]
        elif kind == "scalar":
            columns.append(arr.reshape(-1, 1))  # [N, 1]
        else:
            raise ValueError(f"Unknown kind '{kind}' for field '{name}' — use 'vector' or 'scalar'")

    volume_fields = np.concatenate(columns, axis=1)  # [N, total_channels]

    # Drop any rows with NaN or Inf (e.g. cells inside the geometry)
    valid_mask = np.isfinite(volume_fields).all(axis=1)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"    Dropping {n_invalid} invalid (NaN/Inf) cells")
        volume_fields = volume_fields[valid_mask]
        coords        = coords[valid_mask]

    return {
        "volume_mesh_centers": coords,        # [N_valid, 3]
        "volume_fields":       volume_fields, # [N_valid, total_channels]
    }


def convert_case(
    case_dir: Path,
    output_dir: Path,
    field_names: dict,
    field_location: str,
    inlet_velocity: float,
    air_density: float,
    skip_existing: bool,
) -> bool:
    """Convert one case folder to a zarr file. Returns True on success."""
    out_path = output_dir / f"{case_dir.name}.zarr"

    if skip_existing and out_path.exists():
        print(f"  SKIP (already exists): {out_path.name}")
        return True

    stl_path = case_dir / STL_FILENAME
    vti_path = case_dir / VTI_FILENAME

    if not stl_path.exists():
        print(f"  ERROR: {stl_path} not found — skipping")
        return False
    if not vti_path.exists():
        print(f"  ERROR: {vti_path} not found — skipping")
        return False

    try:
        stl_data = read_stl(stl_path, unit_scale=STL_UNIT_SCALE)
        vti_data = read_vti(vti_path, field_names, field_location)
    except Exception as e:
        print(f"  ERROR reading {case_dir.name}: {e}")
        return False

    # Global parameters: shape [n_params, 1]
    # Order must match variables.global_parameters in config.yaml
    global_params = np.array([[inlet_velocity], [air_density]], dtype=np.float32)

    store = zarr.open(str(out_path), mode="w")
    store["stl_coordinates"]     = stl_data["stl_coordinates"]
    store["stl_centers"]         = stl_data["stl_centers"]
    store["stl_faces"]           = stl_data["stl_faces"]
    store["stl_areas"]           = stl_data["stl_areas"]
    store["volume_mesh_centers"] = vti_data["volume_mesh_centers"]
    store["volume_fields"]       = vti_data["volume_fields"]
    store["global_params_values"]    = global_params
    store["global_params_reference"] = global_params

    n_verts = stl_data["stl_coordinates"].shape[0]
    n_faces = stl_data["stl_centers"].shape[0]
    n_vol   = vti_data["volume_mesh_centers"].shape[0]
    n_ch    = vti_data["volume_fields"].shape[1]
    print(f"  OK: stl={n_verts} verts / {n_faces} faces | vol={n_vol} pts / {n_ch} channels")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not INPUT_DIR.exists():
        print(f"ERROR: INPUT_DIR does not exist: {INPUT_DIR}")
        sys.exit(1)

    # Auto-discover fields if VOLUME_FIELD_NAMES is empty
    if not VOLUME_FIELD_NAMES:
        print("VOLUME_FIELD_NAMES is empty — discovering available fields ...\n")
        discover_fields(INPUT_DIR)
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    print(f"Found {len(case_dirs)} cases in {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Fields: {VOLUME_FIELD_NAMES}")
    print(f"Field location: {FIELD_LOCATION}\n")

    n_ok   = 0
    n_fail = 0
    for case_dir in case_dirs:
        print(f"[{case_dir.name}]")
        success = convert_case(
            case_dir      = case_dir,
            output_dir    = OUTPUT_DIR,
            field_names   = VOLUME_FIELD_NAMES,
            field_location= FIELD_LOCATION,
            inlet_velocity= INLET_VELOCITY,
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
