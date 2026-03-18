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
Split a directory of .zarr cases into train and validation sets.

Validation cases are moved out of zarr_dir into a separate val directory.
zarr_dir itself becomes the training directory — no data is copied for train.

By default every 5th case (sorted order) goes to val. Use --random to
shuffle before splitting, and --val_pct to control the fraction.

Usage:
    python split_zarr.py --zarr_dir /data/zarr

    # Custom val directory:
    python split_zarr.py --zarr_dir /data/zarr --val_dir /data/zarr_val

    # 20% val, random shuffle, reproducible:
    python split_zarr.py --zarr_dir /data/zarr --val_pct 0.2 --random --seed 42

After running, set in config.yaml:
    data:
      input_dir:     /data/zarr       # unchanged, now train-only
      input_dir_val: /data/zarr_val
"""

import argparse
import os
import random
import shutil
import subprocess
from pathlib import Path


def split_zarr(
    zarr_dir: Path,
    val_dir: Path,
    val_pct: float = 0.2,
    randomize: bool = False,
    seed: int = 42,
) -> None:
    """
    Move val cases out of zarr_dir into val_dir.
    zarr_dir is left in place as the training directory.

    Args:
        zarr_dir:  Directory containing all <case_id>.zarr folders.
        val_dir:   Destination for validation cases.
        val_pct:   Fraction of cases assigned to val (default: 0.2 = 20%).
        randomize: If True, shuffle cases before splitting.
        seed:      Random seed for reproducibility (only used when randomize=True).
    """
    cases = sorted(
        [d for d in zarr_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")],
        key=lambda d: int(d.stem) if d.stem.isdigit() else d.stem,
    )

    if not cases:
        print(f"ERROR: No .zarr directories found in {zarr_dir}")
        return

    print(f"Found {len(cases)} total cases")

    if randomize:
        rng = random.Random(seed)
        rng.shuffle(cases)
        print(f"Shuffled with seed={seed}")

    n_val = max(1, round(len(cases) * val_pct))
    val_cases = cases[:n_val]

    print(f"Train: {len(cases) - n_val} cases  ({100*(1-val_pct):.0f}%)")
    print(f"Val:   {n_val} cases  ({100*val_pct:.0f}%)")
    print(f"Val cases: {[c.name for c in val_cases]}")

    val_dir.mkdir(parents=True, exist_ok=True)

    for case in val_cases:
        dst = val_dir / case.name
        if dst.exists():
            print(f"  SKIP (exists): {case.name}")
            continue
        try:
            os.rename(case, dst)
        except OSError:
            print(
                f"  WARNING: cannot rename {case.name} — falling back to cp -r "
                "(slow for large datasets)."
            )
            subprocess.run(["cp", "-r", str(case), str(dst)], check=True)
            shutil.rmtree(str(case))
        print(f"  VAL <- {case.name}")

    print(f"\nDone.")
    print(f"  input_dir:     {zarr_dir}")
    print(f"  input_dir_val: {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move val cases out of zarr_dir into a separate val directory."
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        required=True,
        help="Directory containing all .zarr case folders (becomes train dir)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Destination for validation cases (default: <zarr_dir>/../zarr_val)",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=0.2,
        help="Fraction of cases assigned to val (default: 0.2)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle cases randomly before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42, only used with --random)",
    )
    args = parser.parse_args()

    zarr_dir = Path(args.zarr_dir)
    val_dir  = Path(args.val_dir) if args.val_dir else zarr_dir.parent / "zarr_val"

    split_zarr(
        zarr_dir=zarr_dir,
        val_dir=val_dir,
        val_pct=args.val_pct,
        randomize=args.random,
        seed=args.seed,
    )

