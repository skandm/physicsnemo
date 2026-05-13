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
#
# K8s-ready image for PhysicsNeMo training/inference.
# Targets linux/amd64 only (no ARM handling).
# Source tree is preserved at /physicsnemo so scripts/examples/configs are accessible.
# wandb and mlflow are retained for experiment tracking.
#
# Smoke test:
#   docker run --rm --gpus all <image> \
#     python -c "import physicsnemo; import wandb; import torch; print(torch.cuda.is_available())"

FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Update pip and setuptools
RUN uv pip install "pip>=23.2.1" "setuptools>=77.0.3"

# System dependencies
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 zip unzip && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Remove packaging==23.2 from constraint.txt if present
RUN FILE="/etc/pip/constraint.txt" && \
    if [ -f "$FILE" ]; then \
        sed -i '/packaging/d' "$FILE"; \
    else \
        echo "File not found: $FILE"; \
    fi

# Respect the container's constraint file; create empty one if missing
RUN [ -f /etc/pip/constraint.txt ] || touch /etc/pip/constraint.txt
ENV UV_CONSTRAINT=/etc/pip/constraint.txt

# Clone physicsnemo source from public repo (branch dev/sk/ai)
RUN git clone --branch dev/sk/ai --single-branch \
    https://github.com/skandm/physicsnemo.git /physicsnemo

# amd64-only deps: pyspng, numcodecs
RUN uv pip install "pyspng>=0.1.0"
RUN uv pip install numcodecs

# vtk + pyvista
RUN uv pip install "vtk>=9.2.6"
RUN uv pip install "pyvista>=0.40.1"

# onnxruntime-gpu
RUN uv pip install "onnxruntime-gpu>1.19.0"

# torch-geometric
RUN uv pip install "torch_geometric>=2.6.1"

# CUDA arch list used when building extensions from source
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX"

# torch_scatter — build from source (no pre-built wheel assumed)
RUN mkdir -p /physicsnemo/deps/ && \
    cd /physicsnemo/deps/ && \
    git clone https://github.com/rusty1s/pytorch_scatter.git && \
    cd pytorch_scatter && \
    git checkout tags/2.1.2 && \
    FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
    uv pip install --reinstall dist/*.whl && \
    cd ../ && rm -rf pytorch_scatter

# pyg-lib — build from source
RUN uv pip install ninja wheel && \
    uv pip install --no-build-isolation "git+https://github.com/pyg-team/pyg-lib.git@0.5.0"

# torch_cluster — build from source
RUN mkdir -p /physicsnemo/deps/ && \
    cd /physicsnemo/deps/ && \
    git clone --branch 1.6.3 --depth 1 https://github.com/rusty1s/pytorch_cluster.git && \
    cd pytorch_cluster && \
    FORCE_CUDA=1 MAX_JOBS=64 python setup.py bdist_wheel && \
    uv pip install --reinstall dist/*.whl && \
    cd ../ && rm -rf pytorch_cluster

# natten — build from source
ENV NATTEN_CUDA_ARCH="8.0;8.6;9.0;10.0;12.0"
RUN mkdir -p /physicsnemo/deps/ && \
    cd /physicsnemo/deps/ && \
    git clone --recursive --branch v0.21.5 --depth 1 https://github.com/SHI-Labs/NATTEN.git && \
    cd NATTEN && \
    MAX_JOBS=64 python setup.py bdist_wheel && \
    uv pip install --reinstall dist/*.whl && \
    cd ../ && rm -rf NATTEN

# torch_sparse (needs torch at build time)
RUN uv pip install --no-build-isolation "torch_sparse"

# Install physicsnemo with all relevant extras (wandb/mlflow kept via utils-extras)
RUN cd /physicsnemo && \
    uv pip install ".[cu13,utils-extras,mesh-extras,datapipes-extras,gnns,perf]"

# Clean uv cache to reduce image size
RUN uv cache clean

WORKDIR /physicsnemo

ENV PYTHONUNBUFFERED=1
ENV _CUDA_COMPAT_TIMEOUT=90
