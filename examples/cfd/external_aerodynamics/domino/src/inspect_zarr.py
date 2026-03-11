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
Inspects and visualizes the statistical distribution of all fields
across a directory of .zarr or .npy files.

Uses streaming statistics (Welford's algorithm) and reservoir sampling so
memory usage stays constant regardless of dataset size — safe for hundreds
of large zarr files without OOM kills.

Useful for verifying data quality before training:
  - Check velocity/pressure ranges look physically reasonable
  - Spot outliers or unit conversion issues
  - Understand coordinate extents for setting bounding_box in config.yaml

Usage:
    conda activate domino
    python inspect_zarr.py --data_dir /path/to/zarr_data

    # Limit to specific fields (faster):
    python inspect_zarr.py --data_dir /path/to/zarr_data --fields volume_fields stl_areas

    # Limit how many cases to sample (faster for large datasets):
    python inspect_zarr.py --data_dir /path/to/zarr_data --max_cases 10

    # Save plots to disk instead of showing interactively:
    python inspect_zarr.py --data_dir /path/to/zarr_data --save_dir /path/to/plots
"""

import argparse
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import zarr

# Read zarr arrays in chunks of this many rows to avoid loading huge arrays at once.
_CHUNK_ROWS = 200_000

# Maximum number of values kept in the reservoir sample (used for histograms/box plots).
_RESERVOIR_SIZE = 50_000


class _StreamingStats:
    """
    Accumulates statistics over an arbitrarily large stream of float values
    without storing all of them.

    - min / max / mean / std are exact (Welford's parallel algorithm).
    - percentiles and histogram use a reservoir sample of fixed size.
    """

    def __init__(self):
        self.n = 0
        self._min = np.inf
        self._max = -np.inf
        self._mean = 0.0
        self._M2 = 0.0
        self._reservoir = np.empty(_RESERVOIR_SIZE, dtype=np.float32)
        self._res_filled = 0

    def update(self, arr):
        """Incorporate a numpy array (any shape) into the running statistics."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        n_new = len(arr)
        if n_new == 0:
            return

        # --- exact stats ---
        self._min = min(self._min, float(arr.min()))
        self._max = max(self._max, float(arr.max()))

        # Welford's parallel (Chan et al.) algorithm
        n_old = self.n
        mean_new = float(arr.mean())
        M2_new = float(np.sum((arr - mean_new) ** 2))
        n_total = n_old + n_new
        delta = mean_new - self._mean
        self._mean += delta * n_new / n_total
        self._M2 += M2_new + delta**2 * n_old * n_new / n_total
        self.n = n_total

        # --- reservoir sampling (Algorithm R) ---
        # Fill unfilled slots first
        if self._res_filled < _RESERVOIR_SIZE:
            take = min(_RESERVOIR_SIZE - self._res_filled, n_new)
            self._reservoir[self._res_filled : self._res_filled + take] = arr[
                :take
            ].astype(np.float32)
            self._res_filled += take
            arr = arr[take:]
            n_old += take
            n_new = len(arr)
            if n_new == 0:
                return

        # Vectorised replacement for remaining elements
        global_pos = np.arange(n_old, n_old + n_new, dtype=np.int64)
        replace_prob = _RESERVOIR_SIZE / (global_pos + 1)
        mask = np.random.rand(n_new) < replace_prob
        if mask.any():
            target = np.random.randint(0, _RESERVOIR_SIZE, n_new)
            self._reservoir[target[mask]] = arr[mask].astype(np.float32)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return float(np.sqrt(self._M2 / (self.n - 1))) if self.n > 1 else 0.0

    def percentile(self, q):
        s = self._reservoir[: self._res_filled]
        return float(np.percentile(s, q)) if len(s) else float("nan")

    def sample(self):
        return self._reservoir[: self._res_filled]


def analyze_and_plot_distribution(stats, field_name, save_dir=None):
    """Print statistics and plot the distribution for one field."""
    if stats.n == 0:
        print(f"\n--- No data found for '{field_name}'. Skipping. ---\n")
        return

    print(f"\n{'---'*5} Analysis for: {field_name.upper()} {'---'*5}")
    print(f"Total values: {stats.n:,}")
    print(
        f"Statistics:\n"
        f"  Min:     {stats._min:.4f}\n"
        f"  Max:     {stats._max:.4f}\n"
        f"  Mean:    {stats.mean:.4f}\n"
        f"  Std Dev: {stats.std:.4f}"
    )

    print("\n--- Percentile Distribution ---")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"  {p:2d}th percentile: {stats.percentile(p):.4f}")
    print("*" * 40)

    sample = stats.sample()
    if len(sample) == 0:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        f"Distribution for: {field_name}  (n={stats.n:,}, sample={len(sample):,})",
        fontsize=16,
        y=0.98,
    )

    ax1.hist(sample, bins=100, color="skyblue", edgecolor="black", alpha=0.8)
    ax1.set_title("Histogram (reservoir sample)")
    ax1.set_xlabel(field_name)
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax2.boxplot(
        sample,
        vert=False,
        whis=[5, 95],
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen"),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=5, alpha=0.3),
    )
    ax2.set_title("Box Plot (whiskers = 5th–95th percentile, red = outliers)")
    ax2.set_xlabel(field_name)
    ax2.set_yticks([])
    ax2.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = field_name.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(save_dir, f"{safe_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def _update_stats(key, array, stats_dict):
    """
    Feed one (possibly 2-D) numpy chunk into the appropriate StreamingStats buckets.
    Splits 2-D arrays by column; computes velocity magnitude for vector fields.
    """
    array = np.asarray(array)
    if array.size == 0:
        return

    if array.ndim == 1:
        stats_dict[key].update(array)

    elif array.ndim == 2:
        for i in range(array.shape[1]):
            stats_dict[f"{key}_col_{i}"].update(array[:, i])
        if ("velo" in key.lower() or "field" in key.lower()) and array.shape[1] >= 3:
            stats_dict[f"{key}_Magnitude"].update(
                np.linalg.norm(array[:, :3], axis=1)
            )

    else:
        # Flatten higher-dimensional arrays
        stats_dict[key].update(array.ravel())


def _zarr_walk(obj, stats_dict, prefix=""):
    """Recursively walk a zarr Group/Array, feeding data in chunks."""
    if isinstance(obj, zarr.Group):
        for k in obj:
            new_prefix = f"{prefix}_{k}" if prefix else k
            _zarr_walk(obj[k], stats_dict, new_prefix)

    elif isinstance(obj, zarr.Array):
        n_rows = obj.shape[0] if obj.ndim >= 1 else 1
        for start in range(0, n_rows, _CHUNK_ROWS):
            chunk = obj[start : start + _CHUNK_ROWS]
            _update_stats(prefix, chunk, stats_dict)


def process_and_plot_directory(
    data_dir,
    fields_filter=None,
    max_cases=None,
    save_dir=None,
):
    """
    Load all .zarr dirs and .npy files, accumulate streaming statistics,
    then plot the distribution of each field.

    Args:
        data_dir:      Path to directory containing .zarr folders or .npy files.
        fields_filter: If provided, only plot fields whose names contain one of
                       these strings. e.g. ["volume_fields", "stl_areas"]
        max_cases:     If set, only process this many cases (useful for quick checks).
        save_dir:      If set, save plots as PNG files here instead of showing them.
    """
    stats_dict = defaultdict(_StreamingStats)

    npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    zarr_dirs = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".zarr") and os.path.isdir(os.path.join(data_dir, f))
        ]
    )

    if not npy_files and not zarr_dirs:
        print(f"ERROR: No .npy files or .zarr directories found in '{data_dir}'.")
        return

    print(f"Found {len(npy_files)} .npy files and {len(zarr_dirs)} .zarr directories.")

    if max_cases is not None:
        all_sources = npy_files + zarr_dirs
        all_sources = all_sources[:max_cases]
        npy_files = [s for s in all_sources if s.endswith(".npy")]
        zarr_dirs = [s for s in all_sources if s.endswith(".zarr")]
        print(f"Limiting to {max_cases} cases.")

    # --- .npy files ---
    for file_path in npy_files:
        try:
            data_dict = np.load(file_path, allow_pickle=True).item()
            for key, array in data_dict.items():
                if isinstance(array, np.ndarray):
                    _update_stats(key, array, stats_dict)
            print(f"  Processed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(file_path)}: {e}")

    # --- .zarr directories ---
    for zarr_path in zarr_dirs:
        try:
            root = zarr.open(zarr_path, mode="r")
            _zarr_walk(root, stats_dict)
            print(f"  Processed: {os.path.basename(zarr_path)}")
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(zarr_path)}: {e}")

    if not stats_dict:
        print("No valid data found. Exiting.")
        return

    # --- field filter ---
    if fields_filter:
        filtered = {
            k: v
            for k, v in stats_dict.items()
            if any(f.lower() in k.lower() for f in fields_filter)
        }
        if not filtered:
            print(f"WARNING: No fields matched filter {fields_filter}.")
            print(f"Available fields: {list(stats_dict.keys())}")
            return
        stats_dict = filtered

    print(f"\nGenerating plots for {len(stats_dict)} fields...\n")
    for field_name, stats in sorted(stats_dict.items()):
        analyze_and_plot_distribution(stats, field_name, save_dir=save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and plot field distributions from .zarr / .npy datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .zarr folders or .npy files",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="Only plot fields whose names contain these strings. "
        "e.g. --fields volume_fields stl_areas",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Only process this many cases (useful for a quick check)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Save plots as PNG files to this directory instead of showing them",
    )
    args = parser.parse_args()

    process_and_plot_directory(
        data_dir=args.data_dir,
        fields_filter=args.fields,
        max_cases=args.max_cases,
        save_dir=args.save_dir,
    )
