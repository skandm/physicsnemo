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
Compute and save scaling factors for DoMINO datasets.

Computes mean, std, min, and max for all field variables across the training
zarr dataset. The output scaling_factors.pkl is required by both training
and inference.

Usage:
    python compute_statistics.py --data_dir /data/zarr_train --output /data/scaling_factors/scaling_factors.pkl

    # Limit samples for faster computation on large datasets:
    python compute_statistics.py --data_dir /data/zarr_train --output /data/scaling_factors/scaling_factors.pkl --max_samples 500

    # Force recompute even if scaling_factors.pkl already exists:
    python compute_statistics.py --data_dir /data/zarr_train --output /data/scaling_factors/scaling_factors.pkl --force
"""

# Warp 1.x compatibility shim: physicsnemo uses wp.context.Device as a type
# annotation which was removed in warp 1.0. Recreate it as a lightweight module.
import warp as wp
import types as _types, sys as _sys
if not hasattr(wp, "context"):
    _ctx = _types.ModuleType("warp.context")
    _ctx.Device = object  # used as type annotation only, not at runtime
    wp.context = _ctx
    _sys.modules["warp.context"] = _ctx

import argparse
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import compute_scaling_factors
from utils import ScalingFactors


def main():
    parser = argparse.ArgumentParser(
        description="Compute DoMINO scaling factors from a zarr training dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training .zarr case folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save scaling_factors.pkl (directory will be created if needed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to config.yaml (default: conf/config.yaml)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help=(
            "Max number of data points to sample when computing statistics. "
            "Lower is faster but less accurate. "
            "Defaults to max_samples_for_statistics in config.yaml."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if scaling_factors.pkl already exists",
    )
    args = parser.parse_args()

    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("ComputeStatistics")
    logger = RankZeroLoggingWrapper(logger, dist)

    # Load config and apply CLI overrides
    cfg = OmegaConf.load(args.config)
    cfg.data.input_dir = args.data_dir
    cfg.data.scaling_factors = args.output
    if args.max_samples is not None:
        cfg.data.max_samples_for_statistics = args.max_samples

    logger.info("Starting scaling factors computation")
    logger.info(f"  data_dir:    {cfg.data.input_dir}")
    logger.info(f"  output:      {cfg.data.scaling_factors}")
    logger.info(f"  max_samples: {cfg.data.max_samples_for_statistics}")

    output_dir = os.path.dirname(cfg.data.scaling_factors)
    os.makedirs(output_dir, exist_ok=True)

    if dist.world_size > 1:
        torch.distributed.barrier()

    pickle_path = cfg.data.scaling_factors

    if not args.force and Path(pickle_path).exists():
        logger.info(f"Scaling factors already exist at: {pickle_path}")
        logger.info("Use --force to recompute.")
        return

    logger.info("Computing scaling factors from dataset...")
    start_time = time.perf_counter()

    model_type = cfg.model.model_type
    target_keys = ["stl_centers"]
    if model_type in ("volume", "combined"):
        target_keys += ["volume_mesh_centers", "volume_fields"]
    if model_type in ("surface", "combined"):
        target_keys += ["surface_mesh_centers", "surface_fields"]

    mean, std, min_val, max_val = compute_scaling_factors(
        cfg=cfg,
        input_path=cfg.data.input_dir,
        target_keys=target_keys,
        max_samples=cfg.data.max_samples_for_statistics,
    )
    mean    = {k: m.cpu().numpy() for k, m in mean.items()}
    std     = {k: s.cpu().numpy() for k, s in std.items()}
    min_val = {k: m.cpu().numpy() for k, m in min_val.items()}
    max_val = {k: m.cpu().numpy() for k, m in max_val.items()}

    compute_time = time.perf_counter() - start_time
    logger.info(f"Computation completed in {compute_time:.2f}s")

    scaling_factors = ScalingFactors(
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        field_keys=target_keys,
    )

    if dist.rank == 0:
        scaling_factors.save(pickle_path)
        logger.info(f"Scaling factors saved to: {pickle_path}")

        summary_path = str(Path(pickle_path).parent / "scaling_factors_summary.txt")
        with open(summary_path, "w") as f:
            f.write(scaling_factors.summary())
        logger.info(f"Summary saved to: {summary_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
