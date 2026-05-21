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

import torch

from .utils import validate_inputs


def knn_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform kNN search with torch.

    Args:
        points (torch.Tensor): Query points, shape (M, D)
        queries (torch.Tensor): Reference points, shape (N, D)
        k (int): Number of neighbors

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - indices (torch.Tensor): Indices of the top-k nearest neighbors, shape (N, k)
            - distances (torch.Tensor): Distances to the top-k nearest neighbors, shape (N, k)
    """
    validate_inputs(points, queries)
    # M, D = p1.shape
    # N, D_feat = p2_features.shape

    # Compute pairwise distances: (M, N)
    dists = torch.norm(points[:, None, :] - queries[None, :, :], dim=-1)

    n_points = dists.shape[0]
    n_queries = dists.shape[1]

    if n_points == 0:
        # No reference points — return zero-filled tensors of the expected shape
        idx = torch.zeros((n_queries, k), dtype=torch.long, device=points.device)
        dist = torch.zeros((n_queries, k), dtype=points.dtype, device=points.device)
        return idx, dist

    # Find top-k nearest neighbors (clamp k to available points)
    k_eff = min(k, n_points)
    topk_dists, topk_idx = torch.topk(dists, k=k_eff, dim=0, largest=False, sorted=True)

    idx_out = topk_idx.T    # (n_queries, k_eff)
    dist_out = topk_dists.T # (n_queries, k_eff)

    # Pad to (n_queries, k) by repeating the last valid neighbor
    if k_eff < k:
        pad = k - k_eff
        idx_out = torch.cat([idx_out, idx_out[:, -1:].expand(-1, pad)], dim=1)
        dist_out = torch.cat([dist_out, dist_out[:, -1:].expand(-1, pad)], dim=1)

    return idx_out, dist_out
