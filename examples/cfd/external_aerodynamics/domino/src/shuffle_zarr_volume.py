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
Copy a zarr training directory to a new location with volume data shuffled.

WHY: The DoMINO datapipe's volume_sample_from_disk mode reads contiguous chunks
from zarr and assumes the data is pre-shuffled. If the volume points are stored
in CFD mesh order (spatially correlated), contiguous chunks can land in a single
spatial region that may fall entirely outside the bounding box, causing:

    ValueError: Volume mesh has fewer points than requested sample size

This script creates a new copy of the dataset where volume_mesh_centers and
volume_fields are randomly permuted, so contiguous reads are spatially random.

What is shuffled (same permutation applied to both to keep them aligned):
    volume_mesh_centers  [N, 3]  — volume point coordinates
    volume_fields        [N, C]  — field values (Vx, Vy, Vz, P, ...)

What is copied as-is (not row-aligned with volume data):
    stl_coordinates      [V, 3]  — STL vertex coordinates
    stl_centers          [F, 3]  — STL face centers
    stl_faces            [F*3]   — STL face vertex indices
    stl_areas            [F]     — STL face areas
    global_params_values         — scalar global parameters
    global_params_reference      — scalar global parameters

Usage:
    # Basic: copy zarr_data → zarr_data_shuffled
    python shuffle_zarr_volume.py \\
        --src_dir /data/zarr_data \\
        --dst_dir /data/zarr_data_shuffled

    # Dry run (show what would be done without writing anything):
    python shuffle_zarr_volume.py \\
        --src_dir /data/zarr_data \\
        --dst_dir /data/zarr_data_shuffled \\
        --dry_run

    # Resume interrupted run (skip already completed cases):
    python shuffle_zarr_volume.py \\
        --src_dir /data/zarr_data \\
        --dst_dir /data/zarr_data_shuffled \\
        --skip_done

    # Reproducible shuffle with fixed seed:
    python shuffle_zarr_volume.py \\
        --src_dir /data/zarr_data \\
        --dst_dir /data/zarr_data_shuffled \\
        --seed 42

After running, update config.yaml:
    data:
      input_dir: /data/zarr_data_shuffled   # shuffled training data
      volume_sample_from_disk: true         # safe to re-enable
"""

import argparse
import time
from pathlib import Path

import numpy as np
import zarr

# Keys that contain per-point volume data and must be shuffled together.
VOLUME_KEYS = ["volume_mesh_centers", "volume_fields"]

# Keys that are copied verbatim (not row-aligned with volume data).
PASSTHROUGH_KEYS = [
    "stl_coordinates",
    "stl_centers",
    "stl_faces",
    "stl_areas",
    "global_params_values",
    "global_params_reference",
]

# Attribute written to each output zarr to mark it as shuffled.
MARKER_ATTR = "volume_shuffled"


def copy_case(
    src_path: Path,
    dst_path: Path,
    rng: np.random.Generator,
    dry_run: bool,
) -> bool:
    """
    Copy one zarr case to dst_path with volume arrays shuffled.

    Returns True on success, False on error.
    """
    try:
        src = zarr.open(str(src_path), mode="r")
    except Exception as e:
        print(f"  ERROR opening source {src_path.name}: {e}")
        return False

    # Determine volume size for progress reporting
    if "volume_mesh_centers" not in src:
        print(f"  ERROR: volume_mesh_centers not found in {src_path.name}")
        return False

    n_points = src["volume_mesh_centers"].shape[0]
    n_fields = src["volume_fields"].shape[1] if "volume_fields" in src else "?"
    print(
        f"  {src_path.name}: {n_points:,} volume pts, {n_fields} field channels",
        end="",
        flush=True,
    )

    if dry_run:
        print("  [dry_run]")
        return True

    t0 = time.perf_counter()

    # Create destination zarr
    try:
        dst = zarr.open(str(dst_path), mode="w")
    except Exception as e:
        print(f"\n  ERROR creating destination {dst_path}: {e}")
        return False

    # ── 1. Generate volume permutation ────────────────────────────────────────
    idx = rng.permutation(n_points)

    # ── 2. Shuffle and write volume arrays ────────────────────────────────────
    for key in VOLUME_KEYS:
        if key not in src:
            print(f"\n  WARNING: {key} not found in {src_path.name}, skipping")
            continue
        data = src[key][:]       # load full array
        data = data[idx]         # apply permutation
        dst[key] = data
        del data

    # ── 3. Copy passthrough arrays verbatim ───────────────────────────────────
    for key in PASSTHROUGH_KEYS:
        if key not in src:
            # Not all keys are required (e.g. surface keys absent in volume-only)
            continue
        dst[key] = src[key][:]

    # ── 4. Copy any extra keys not in either list (future-proofing) ───────────
    known_keys = set(VOLUME_KEYS + PASSTHROUGH_KEYS)
    for key in src.keys():
        if key not in known_keys:
            print(f"\n  NOTE: copying unknown key '{key}' verbatim")
            dst[key] = src[key][:]

    # ── 5. Copy zarr attributes and add shuffle marker ────────────────────────
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    dst.attrs[MARKER_ATTR] = True

    elapsed = time.perf_counter() - t0
    print(f"  ({elapsed:.1f}s)")
    return True


def shuffle_zarr_volume(
    src_dir: Path,
    dst_dir: Path,
    seed: int,
    dry_run: bool,
    skip_done: bool,
) -> None:
    cases = sorted(
        [d for d in src_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")],
        key=lambda d: d.name,
    )

    if not cases:
        print(f"ERROR: No .zarr directories found in {src_dir}")
        return

    print(f"Source: {src_dir}  ({len(cases)} cases)")
    print(f"Dest:   {dst_dir}")
    if dry_run:
        print("DRY RUN — no files will be written\n")
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        print()

    # Seeded RNG — each case gets a unique deterministic seed
    base_rng = np.random.default_rng(seed)

    n_ok = n_skip = n_fail = 0
    t_total = time.perf_counter()

    for case in cases:
        dst_path = dst_dir / case.name

        # Check skip_done
        if skip_done and dst_path.exists():
            try:
                z = zarr.open(str(dst_path), mode="r")
                if z.attrs.get(MARKER_ATTR, False):
                    print(f"  SKIP (done): {case.name}")
                    n_skip += 1
                    continue
            except Exception:
                pass  # destination exists but can't be opened — overwrite it

        # Each case gets its own deterministic RNG derived from the base seed
        case_seed = int(base_rng.integers(2**31))
        case_rng = np.random.default_rng(case_seed)

        ok = copy_case(case, dst_path, case_rng, dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    elapsed = time.perf_counter() - t_total
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Shuffled: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}")
    if n_fail == 0 and not dry_run:
        print(f"\nUpdate config.yaml:")
        print(f"  data:")
        print(f"    input_dir: {dst_dir}")
        print(f"    volume_sample_from_disk: true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Copy a zarr training directory to a new location with volume "
            "data shuffled, so that volume_sample_from_disk works correctly "
            "with a bounding box filter."
        )
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Source directory containing .zarr case folders (original data)",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Destination directory for shuffled .zarr cases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="Skip cases that already exist in dst_dir with the shuffle marker",
    )
    args = parser.parse_args()

    shuffle_zarr_volume(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        seed=args.seed,
        dry_run=args.dry_run,
        skip_done=args.skip_done,
    )
