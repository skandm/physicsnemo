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

Cases are sorted numerically by name and every Nth case (default: every 5th)
is assigned to the validation set; the remainder go to train.

Usage:
    python split_zarr.py --zarr_dir /data/my_dataset/zarr

    # Custom output directories:
    python split_zarr.py --zarr_dir /data/zarr --train_dir /data/train --val_dir /data/val

    # Change split ratio (every 10th case goes to val):
    python split_zarr.py --zarr_dir /data/zarr --val_every 10

    # Dry-run to preview the split without moving any files:
    python split_zarr.py --zarr_dir /data/zarr --dry_run
"""

import argparse
import shutil
from pathlib import Path


def split_zarr(
    zarr_dir: Path,
    train_dir: Path,
    val_dir: Path,
    val_every: int = 5,
    dry_run: bool = False,
) -> None:
    """
    Move zarr cases from zarr_dir into train_dir and val_dir.

    Args:
        zarr_dir:  Source directory containing <case_id>.zarr folders.
        train_dir: Destination for training cases.
        val_dir:   Destination for validation cases.
        val_every: Every Nth case (by sorted order) goes to val.
        dry_run:   If True, only print what would happen without moving files.
    """
    cases = sorted(
        [d for d in zarr_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")],
        key=lambda d: int(d.stem) if d.stem.isdigit() else d.stem,
    )

    if not cases:
        print(f"ERROR: No .zarr directories found in {zarr_dir}")
        return

    print(f"Found {len(cases)} total cases")

    train_cases = [c for i, c in enumerate(cases) if i % val_every != 0]
    val_cases   = [c for i, c in enumerate(cases) if i % val_every == 0]

    print(f"Train: {len(train_cases)} cases")
    print(f"Val:   {len(val_cases)} cases  (every {val_every}th case)")
    print(f"Val cases: {[c.name for c in val_cases]}")

    if dry_run:
        print("\n[DRY RUN] No files were moved.")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for case in train_cases:
        dst = train_dir / case.name
        if not dst.exists():
            shutil.move(str(case), str(dst))
            print(f"  TRAIN <- {case.name}")
        else:
            print(f"  SKIP (exists): {case.name}")

    for case in val_cases:
        dst = val_dir / case.name
        if not dst.exists():
            shutil.move(str(case), str(dst))
            print(f"  VAL   <- {case.name}")
        else:
            print(f"  SKIP (exists): {case.name}")

    n_train = len(list(train_dir.iterdir()))
    n_val   = len(list(val_dir.iterdir()))
    print(f"\nDone. Train: {n_train} | Val: {n_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split zarr cases into train and validation directories."
    )
    parser.add_argument(
        "--zarr_dir",
        type=str,
        required=True,
        help="Source directory containing <case_id>.zarr folders",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Destination for training cases (default: <zarr_dir>/../zarr_train)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Destination for validation cases (default: <zarr_dir>/../zarr_val)",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=5,
        help="Every Nth case (by sorted order) is assigned to val (default: 5)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview the split without moving any files",
    )
    args = parser.parse_args()

    zarr_dir = Path(args.zarr_dir)
    train_dir = Path(args.train_dir) if args.train_dir else zarr_dir.parent / "zarr_train"
    val_dir   = Path(args.val_dir)   if args.val_dir   else zarr_dir.parent / "zarr_val"

    split_zarr(
        zarr_dir=zarr_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        val_every=args.val_every,
        dry_run=args.dry_run,
    )
