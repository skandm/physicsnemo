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

Useful for verifying data quality before training:
  - Check velocity/pressure ranges look physically reasonable
  - Spot outliers or unit conversion issues
  - Understand coordinate extents for setting bounding_box in config.yaml

Usage:
    conda activate domino
    python inspect_zarr.py --data_dir D:/Downloads/raf_test/zarr

    # Limit to specific fields (faster):
    python inspect_zarr.py --data_dir D:/Downloads/raf_test/zarr --fields volume_fields stl_areas

    # Limit how many cases to sample (faster for large datasets):
    python inspect_zarr.py --data_dir D:/Downloads/raf_test/zarr --max_cases 10

    # Save plots to disk instead of showing interactively:
    python inspect_zarr.py --data_dir D:/Downloads/raf_test/zarr --save_dir D:/Downloads/raf_test/plots
"""

import argparse
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import zarr


def analyze_and_plot_distribution(data_array, field_name, save_dir=None):
    """
    Calculates statistics and plots the distribution for a given 1D data array.
    """
    if data_array.size == 0:
        print(f"\n--- No data found for '{field_name}'. Skipping analysis. ---\n")
        return

    data_array = data_array.flatten()
    print(f"\n{'---'*5} Analysis for: {field_name.upper()} {'---'*5}")
    print(f"Shape of aggregated data: {data_array.shape}")
    print(f"Total values: {len(data_array):,}")

    print(
        f"Statistics:\n"
        f"  Min:     {data_array.min():.4f}\n"
        f"  Max:     {data_array.max():.4f}\n"
        f"  Mean:    {data_array.mean():.4f}\n"
        f"  Std Dev: {data_array.std():.4f}"
    )

    print("\n--- Percentile Distribution ---")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = np.percentile(data_array, percentiles)
    for p, v in zip(percentiles, percentile_values):
        print(f"  {p:2d}th percentile: {v:.4f}")
    print("*" * 40)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"Distribution for: {field_name}", fontsize=16, y=0.98)

    # Histogram
    ax1.hist(data_array, bins=100, color="skyblue", edgecolor="black", alpha=0.8)
    ax1.set_title("Histogram")
    ax1.set_xlabel(field_name)
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Box plot
    ax2.boxplot(
        data_array,
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


def _process_array_data(key, array, aggregated_data):
    """
    Process a single array: split columns and compute velocity magnitude.
    """
    if not isinstance(array, np.ndarray):
        try:
            array = np.array(array)
        except Exception:
            return

    if array.ndim == 1:
        aggregated_data[key].extend(array)

    elif array.ndim == 2:
        num_cols = array.shape[1]
        for i in range(num_cols):
            aggregated_data[f"{key}_col_{i}"].extend(array[:, i])

        # Compute velocity magnitude if this looks like a vector field
        if ("velo" in key.lower() or "field" in key.lower()) and num_cols >= 3:
            magnitudes = np.linalg.norm(array[:, :3], axis=1)
            aggregated_data[f"{key}_Magnitude"].extend(magnitudes)


def process_and_plot_directory(
    data_dir,
    fields_filter=None,
    max_cases=None,
    save_dir=None,
):
    """
    Loads all .zarr dirs and .npy files, aggregates data across all cases,
    and plots the distribution of each field.

    Args:
        data_dir:      Path to directory containing .zarr folders or .npy files.
        fields_filter: If provided, only plot fields whose names contain one of
                       these strings. e.g. ["volume_fields", "stl_areas"]
        max_cases:     If set, only process this many cases (useful for quick checks).
        save_dir:      If set, save plots as PNG files here instead of showing them.
    """
    aggregated_data = defaultdict(list)

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

    # Process .npy files
    for file_path in npy_files:
        try:
            data_dict = np.load(file_path, allow_pickle=True).item()
            for key, array in data_dict.items():
                if isinstance(array, np.ndarray):
                    _process_array_data(key, array, aggregated_data)
            print(f"  Processed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(file_path)}: {e}")

    # Process .zarr directories
    def _recursive_zarr_walk(obj, prefix=""):
        if isinstance(obj, zarr.Group):
            for k in obj:
                new_prefix = f"{prefix}_{k}" if prefix else k
                _recursive_zarr_walk(obj[k], new_prefix)
        elif isinstance(obj, zarr.Array):
            _process_array_data(prefix, obj[:], aggregated_data)

    for zarr_path in zarr_dirs:
        try:
            root = zarr.open(zarr_path, mode="r")
            _recursive_zarr_walk(root)
            print(f"  Processed: {os.path.basename(zarr_path)}")
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(zarr_path)}: {e}")

    if not aggregated_data:
        print("No valid data aggregated. Exiting.")
        return

    # Apply field filter if specified
    if fields_filter:
        filtered = {
            k: v for k, v in aggregated_data.items()
            if any(f.lower() in k.lower() for f in fields_filter)
        }
        if not filtered:
            print(f"WARNING: No fields matched filter {fields_filter}.")
            print(f"Available fields: {list(aggregated_data.keys())}")
            return
        aggregated_data = filtered

    print(f"\nGenerating plots for {len(aggregated_data)} fields...\n")
    for field_name, data_list in sorted(aggregated_data.items()):
        analyze_and_plot_distribution(
            np.array(data_list), field_name, save_dir=save_dir
        )


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
