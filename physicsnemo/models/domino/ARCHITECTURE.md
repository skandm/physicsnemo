# DoMINO Model Architecture

**Full name:** Decomposable Multi-scale Iterative Neural Operator
**Paper:** https://arxiv.org/abs/2501.13350
**Purpose:** Neural surrogate for large-scale CFD simulations (external aerodynamics). Given an STL geometry and flow conditions, predicts surface pressure/shear stress and volume velocity/pressure fields — replacing hours-long Navier-Stokes solvers with seconds of inference.

---

## Overview

The model is structured as three sequential stages:

```
STL geometry (point cloud) + SDF grid
            │
            ▼
    ┌───────────────────┐
    │     STAGE 1       │  Global Geometry Encoding
    │   GeometryRep     │  Converts point cloud → 3D latent voxel grid
    └───────────────────┘
            │
            │  (B, 4, 128, 64, 64) latent grid
            ▼
    ┌───────────────────┐
    │     STAGE 2       │  Local Geometry Extraction
    │ MultiGeometry     │  Per query point: sample the latent grid
    │   Encoding        │  → 160-dim geometry feature vector
    └───────────────────┘
            │
            │  (B, N_points, 160) per-point geometry context
            ▼
    ┌───────────────────┐
    │     STAGE 3       │  Solution Calculation
    │ SolutionCalculator│  Combines position + geometry context
    │ Surface / Volume  │  → physics field predictions
    └───────────────────┘
            │
            ▼
    Surface: (B, N_surf, 4)   [pressure, WSS_x, WSS_y, WSS_z]
    Volume:  (B, N_vol,  5)   [Vx, Vy, Vz, pressure, turbulent viscosity]
```

Two separate `GeometryRep` instances run in parallel — one for the volume bounding box and one for the tighter surface bounding box — each with their own radii tuned for their respective field scales.

---

## Tensor Notation

| Symbol | Meaning | Typical value |
|---|---|---|
| `B` | Batch size | 1–4 |
| `C` | Channels (feature maps per voxel) | 4 (3 STL + 1 SDF) |
| `Nx, Ny, Nz` | Voxel grid resolution | 128, 64, 64 |
| `N_geo` | Number of STL face centers | ~500,000 |
| `N_surf` | Number of surface query points | ~100,000 |
| `N_vol` | Number of volume query points | ~500,000 |
| `K` | Number of neighbors in ball query | varies by radius |

---

## Stage 1: Global Geometry Encoding

**File:** `geometry_rep.py`
**Class:** `GeometryRep`
**Purpose:** Convert the unstructured STL point cloud into a regular 3D latent voxel grid so that 3D CNNs can propagate spatial context across the entire geometry.

### Why this is needed

Neural networks operating directly on point clouds process each point in isolation. In CFD, this is insufficient — pressure at any point is influenced by the entire body shape due to the elliptic nature of the Navier-Stokes equations (disturbances propagate everywhere). Stage 1 solves this by encoding the geometry into a regular grid where 3D convolutions can propagate global context.

### Two parallel encoding paths

#### Path A: STL Ball Query Encoding

Runs independently for each radius scale. Default surface radii (from `config.py`):

```python
surface_radii:              [0.01, 0.05, 1.0]   # meters
surface_neighbors_in_radius: [8,   16,   128]
```

Volume radii are larger because volume fields extend further from the surface:

```python
volume_radii:              [0.1, 0.5, 1.0, 2.5]
volume_neighbors_in_radius: [32,  64,  128, 256]
```

**For each radius `j`, the pipeline is:**

**Step 1 — BQWarp (Ball Query)**
```
Input:  x       (B, N_geo, 3)         — STL face centers
        p_grid  (B, 128, 64, 64, 3)   — uniform voxel centers
Output: k_short (B, 524288, K, 3)     — K neighbor coordinates per voxel
```
For every voxel in the 128×64×64 grid, finds all STL face centers within radius `r` using GPU-accelerated ball query (NVIDIA Warp). Returns the XYZ coordinates of up to K neighbors. Voxels with fewer than K real neighbors are zero-padded. No learnable parameters.

**Step 2 — GeoConvOut MLP**
```
Input:  k_short  (B, 524288, K, 3)
Output: grid     (B, 1, 128, 64, 64)
```
Flattens the K neighbor coordinates per voxel into a vector of `K×3` numbers, passes through a 2-layer MLP (`K×3 → base_neurons=32 → base_neurons/2=16 → 1`) with tanh activation. Each voxel is independently compressed into a single scalar ∈ (-1, 1) summarizing its local surface geometry configuration. Learnable parameters: MLP weights and biases (~1,280 per scale).

The tanh scalar encodes:
- Near 0: no surface within this radius
- Non-zero: surface present in a particular configuration (flat, curved, sharp edge, etc.)

**Step 3 — UNet (residual)**
```
Input:  grid  (B, 1, 128, 64, 64)
Output: grid  (B, 1, 128, 64, 64)
```
A 3-level encoder-decoder with skip connections and max-pool/upsample:
```
(B, 1,  128, 64, 64) → Conv3d → (B, 8,  128, 64, 64)  [skip_1]
                      → MaxPool → (B, 8,   64, 32, 32)
                      → Conv3d → (B, 16,  64, 32, 32)  [skip_2]
                      → MaxPool → (B, 16,  32, 16, 16)
                      → Conv3d → (B, 32,  32, 16, 16)  [skip_3]
                      → MaxPool → (B, 32,  16,  8,  8)  ← bottleneck
                      → Upsample + cat(skip_3) → Conv3d → (B, 16, 32, 16, 16)
                      → Upsample + cat(skip_2) → Conv3d → (B, 8,  64, 32, 32)
                      → Upsample + cat(skip_1) → Conv3d → (B, 1,  128, 64, 64)
```

Applied as a residual: `x = x + UNet(x) / hops`

The encoder propagates information from individual voxels up to global scale at the bottleneck. The decoder distributes that global context back to full resolution. Skip connections restore fine-grained local detail. After this step, each voxel scalar is informed by the entire geometry — a voxel above the rear windshield now carries signal from the front bumper shape.

Gradient checkpointing is enabled to reduce memory at the cost of recomputation.

**Step 4 — Conv3d (geo_processor_out)**
```
Input:  (B, 1, 128, 64, 64)
Output: (B, 1, 128, 64, 64)
Conv3d(1 → 1, kernel_size=3, padding='same')
```
A single 3D convolution blending each voxel with its 26 immediate neighbors. Smooths any upsampling artifacts from the UNet and provides a final learnable calibration step.

**Concatenating scales:**
```
radius 0.01m → (B, 1, 128, 64, 64)  ┐
radius 0.05m → (B, 1, 128, 64, 64)  ├─ cat(dim=1) → (B, 3, 128, 64, 64)
radius 1.0m  → (B, 1, 128, 64, 64)  ┘
```
Each voxel now has 3 independent scalars answering: "what's within 1cm?", "within 5cm?", "within 1m?".

---

#### Path B: SDF Encoding

```
Input:  sdf  (B, 128, 64, 64)   — precomputed signed distance field on the grid
```

The SDF is expanded into a multi-channel tensor:

```python
binary_sdf   = (sdf >= 0) ? 0.0 : 1.0          # inside/outside indicator
scaled_sdf   = sdf / (s + |sdf|)                # for each s in [0.01, 0.02, 0.04]
sdf_x, sdf_y, sdf_z = torch.gradient(sdf, ...)  # surface normal direction

sdf_input = cat([sdf, scaled_sdf×3, binary_sdf, sdf_x, sdf_y, sdf_z])
# → (B, 8, 128, 64, 64)
```

The `scale_sdf` formula `sdf / (s + |sdf|)` is a soft clamp emphasizing near-surface regions. At scaling factor `s=0.01`, a voxel 1cm from the surface maps to 0.5; at 1m it saturates near 1.0. Three different scaling factors give near-surface contrast at three zoom levels.

The SDF gradients encode the **surface normal direction** at every voxel implicitly — the gradient of the SDF always points toward the nearest surface point.

This 8-channel tensor is passed through another UNet:
```
(B, 8, 128, 64, 64) → UNet → (B, 1, 128, 64, 64)
```

---

#### Combining both paths

```python
encoding_g = cat([stl_encoding, sdf_encoding], dim=1)
# (B, 3, 128, 64, 64) + (B, 1, 128, 64, 64) = (B, 4, 128, 64, 64)
```

The STL path provides explicit surface geometry (precise coordinates, sharp features). The SDF path provides smooth implicit geometry (distance, orientation, inside/outside). Together they give a complementary 4-channel representation. An optional cross-attention UNet (`cross_attention=True`) can further mix the channels.

**Stage 1 output: `(B, 4, 128, 64, 64)`** — a 4-channel 3D voxel grid where each voxel holds learned geometry features summarizing the full body shape.

---

## Stage 2: Local Geometry Extraction

**File:** `encodings.py`
**Classes:** `MultiGeometryEncoding`, `LocalGeometryEncoding`
**Purpose:** For each physics query point (surface triangle center or volume sample), extract a per-point geometry feature vector by sampling the Stage 1 latent grid.

### Why this is needed

Stage 1 produced a grid indexed by voxel position. To make predictions at arbitrary query points (which don't lie on grid nodes), we need to extract geometry context from the grid at each point's location. This is the "decomposable" property: Stage 1 runs once per geometry; Stage 2 just does lookups into that grid — O(N_points), independent of geometry complexity.

### Pipeline (per radius scale)

Stage 2 uses its own set of radii, separate from Stage 1. Default surface radii:

```python
surface_radii:              [0.05, 0.25]
surface_neighbors_in_radius: [32,   128]
```

For each radius, `LocalGeometryEncoding` runs:

**Step 1 — BQWarp on the latent grid**
```
Input:  query points  (B, N_points, 3)    — surface or volume query point positions
        p_grid        (B, 524288, 3)       — voxel centers flattened to point cloud
Output: mapping       (B, N_points, K)    — indices of K nearest voxels per query point
```
For each query point, finds K nearest voxel centers within the radius. Returns voxel indices (not coordinates) for the lookup step.

**Step 2 — Sample the latent grid**
```python
for j in range(4):   # for each of 4 channels from Stage 1
    geo_encoding = reshape(encoding_g[:, j], (B, 1, 524288))
    sampled = index_select(geo_encoding, dim=2, index=mapping.flatten())
    sampled = reshape(sampled, (B, N_points, K)) * mask
```
Uses the voxel indices from Step 1 to gather scalar values from the Stage 1 grid. For each query point, collects K scalars per channel → `K×4` values total. Zero-pads missing neighbors via the mask.

For radius=0.05m with K=32: result is `(B, N_points, 128)` (32 neighbors × 4 channels)

**Step 3 — LocalPointConv MLP**
```
Input:  (B, N_points, K×4)   e.g. (B, N_points, 128)
MLP:    128 → 512 → 32
Output: (B, N_points, 32)
```
Compresses the gathered voxel scalars into a compact feature vector. Learnable parameters: MLP weights and biases.

**Concatenating scales:**
```
radius 0.05m → (B, N_points, 32)   ┐
radius 0.25m → (B, N_points, 128)  ┘ cat(dim=-1) → (B, N_points, 160)
```

**Stage 2 output: `(B, N_points, 160)`** — every query point has a 160-dimensional geometry context vector encoding what the body shape looks like at two spatial scales around it. Because Stage 1 baked global context into the grid, these 160 features implicitly carry information about the entire body, not just the immediate neighborhood.

---

## Stage 3: Solution Calculation

**File:** `solutions.py`
**Classes:** `SolutionCalculatorSurface`, `SolutionCalculatorVolume`
**Purpose:** Combine positional information, geometry context, and basis functions to predict actual physics field values at each query point.

### Three input streams per query point

**Stream 1 — Basis functions (`nn_basis[f]`)**

A FourierMLP applied to the raw point coordinates (+ surface normals + log area for surface points).

*FourierMLP:* Before the MLP, input coordinates are transformed into sine/cosine features at multiple frequencies:
```
x_raw = [x, y, z]
x_fourier = [sin(ω₁x), cos(ω₁x), sin(ω₁y), cos(ω₁y), ..., sin(ω₅z), cos(ω₅z)]
x_input = cat([x_raw, x_fourier])   # 3 + 3×2×5 = 33 values
```
Frequencies are exponentially spaced: `ω = exp(linspace(0, π, 5))`.

This solves spectral bias — plain MLPs struggle to learn high-frequency spatial patterns. By pre-computing sine/cosine basis functions, the MLP only needs to learn coefficients to combine them, enabling sharp spatial transitions (e.g. pressure gradients at leading edges) to be represented accurately.

Each output variable gets its **own independent** `nn_basis[f]` — pressure and shear stress have fundamentally different spatial patterns and benefit from separate representations.

For surface, additional inputs are appended:
- Surface normals `(3,)` — which direction the surface faces (critical for direction-dependent shear stress)
- `log(area) / 10` — triangle size, needed for area-weighted force integration

**Stream 2 — Positional encoding (`encoding_node`)**

Pre-computed in `model.py` before Stage 3. For volume points (when `use_sdf_in_basis_func=True`):
- SDF value at the point
- SDF at multiple scales
- Vector from point to nearest surface point
- Vector from point to geometry center of mass

For surface points:
- Vector from geometry center of mass to the surface point

This tells the model *where the point is relative to the body* — two points with identical geometry context but different positions in the flow field will see different physics.

**Stream 3 — Geometry context (`encoding_g`)**

The 160-dimensional feature vector from Stage 2 — what the geometry looks like around this point.

### Aggregation MLP

All three streams are concatenated and passed through the aggregation MLP:

```python
output = cat([basis_f, encoding_node, encoding_g], dim=-1)
scalar = aggregation_model[f](output)
# MLP: input → 1024 → 1024 → 1024 → 1024 → 1   (GELU activation)
```

Output is **one scalar per output variable per point**. Each variable has its own independent `aggregation_model[f]`.

### Neighbor stencil with inverse-distance weighting

Rather than predicting only at the query point, Stage 3 also predicts at neighboring points and blends results. This acts as a learned finite-difference scheme, forcing spatial consistency.

**For surface** — uses precomputed KNN mesh neighbors (`num_sample_points=7`):
```python
for p in range(num_sample_points):
    if p == 0:
        point = surface_mesh_centers       # the query point itself
    else:
        point = surface_mesh_neighbors[:, :, p-1]   # a mesh neighbor
    prediction[p] = aggregation_model[f](cat([nn_basis[f](point), encoding_node, encoding_g]))
```

**For volume** — randomly samples points in a sphere of radius `1/noise_intensity`:
```python
neighbors = sample_sphere(volume_mesh_centers, radius=1/noise_intensity, num_points=K)
```

**Blending:**
```python
output = 0.5 * prediction_at_center  +  0.5 * Σ(prediction_i / dist_i) / Σ(1 / dist_i)
```
Closer neighbors have higher weight. The 50/50 split between center and distance-weighted average ensures the center point dominates while neighbors provide smoothing.

### Per-variable independent prediction

```python
for f in range(num_variables):
    # Each variable runs through its own nn_basis[f] and aggregation_model[f]
    output_f = stencil_predict(f)   # → (B, N_points, 1)

output_all = cat([output_0, output_1, ..., output_f], dim=-1)
```

**Stage 3 output:**
```
Surface: (B, N_surf, 4)   [pressure, WSS_x, WSS_y, WSS_z]
Volume:  (B, N_vol,  5)   [Vx, Vy, Vz, pressure, turbulent viscosity]
```

---

## Complete Data Flow

```
INPUT
├── geometry_coordinates  (B, N_geo, 3)       STL face centers
├── grid                  (B, 128, 64, 64, 3) Volume voxel centers
├── surf_grid             (B, 128, 64, 64, 3) Surface voxel centers
├── sdf_grid              (B, 128, 64, 64)    SDF on volume grid
├── sdf_surf_grid         (B, 128, 64, 64)    SDF on surface grid
├── sdf_nodes             (B, N_vol, 1)       SDF at volume query points
├── pos_volume_closest    (B, N_vol, 3)       Vector to nearest surface
├── pos_volume_center_of_mass (B, N_vol, 3)  Vector to geometry centroid
├── pos_surface_center_of_mass (B, N_surf, 3)
├── surface_mesh_centers  (B, N_surf, 3)      Surface query point positions
├── surface_mesh_neighbors (B, N_surf, K, 3) KNN neighbors on surface mesh
├── surface_normals       (B, N_surf, 3)
├── surface_areas         (B, N_surf, 1)
├── volume_mesh_centers   (B, N_vol, 3)       Volume query point positions
└── global_params_values  (B, 2, 1)           [inlet_velocity, air_density]

                        │
                        ▼
            ┌─────────────────────────────┐
            │         STAGE 1             │
            │  geo_rep_surface            │
            │  geo_rep_volume             │
            │                             │
            │  For each grid:             │
            │  ├─ BQWarp (per radius)     │
            │  ├─ GeoConvOut MLP          │
            │  ├─ UNet (residual)         │
            │  ├─ Conv3d                  │
            │  └─ SDF UNet               │
            └─────────────────────────────┘
                        │
              surf_encoding (B, 4, 128, 64, 64)
              vol_encoding  (B, 4, 128, 64, 64)
                        │
                        ▼
            ┌─────────────────────────────┐
            │         STAGE 2             │
            │  MultiGeometryEncoding      │
            │  (surface + volume)         │
            │                             │
            │  For each query point:      │
            │  ├─ BQWarp on latent grid   │
            │  ├─ Index into grid         │
            │  └─ LocalPointConv MLP      │
            └─────────────────────────────┘
                        │
              encoding_g_surf (B, N_surf, 160)
              encoding_g_vol  (B, N_vol,  160)
                        │
                        ▼
            ┌─────────────────────────────┐
            │         STAGE 3             │
            │  SolutionCalculatorSurface  │
            │  SolutionCalculatorVolume   │
            │                             │
            │  Per variable f:            │
            │  ├─ FourierMLP(xyz)         │
            │  ├─ cat(basis, node, geo)   │
            │  ├─ AggregationMLP → scalar │
            │  └─ Neighbor stencil blend  │
            └─────────────────────────────┘
                        │
                        ▼
OUTPUT
├── surface_output  (B, N_surf, 4)   [pressure, WSS_x, WSS_y, WSS_z]
└── volume_output   (B, N_vol,  5)   [Vx, Vy, Vz, pressure, turb. viscosity]
```

---

## Key Architecture Components Summary

| Component | Architecture | Role |
|---|---|---|
| `BQWarp` | GPU ball query (NVIDIA Warp) | Spatial neighbor search, no learnable params |
| `GeoConvOut` | 2-layer MLP + tanh | Compress K neighbor coords → 1 scalar per voxel |
| `GeoProcessor` | 3-level encoder-decoder CNN | Alternative to UNet for grid processing |
| `UNet` | 3-depth encoder-decoder, skip connections, optional attention | Propagate spatial context across the full grid |
| `LocalPointConv` | 2-layer MLP (input → 512 → output) | Compress sampled voxel scalars → per-point feature |
| `FourierMLP` | Fourier feature encoding + MLP | High-frequency spatial basis functions |
| `AggregationModel` | 5-layer MLP (4×1024 hidden, GELU) | Final physics prediction per variable |

---

## Configuration Reference (Defaults)

```python
# Grid resolution
interp_res: [128, 64, 64]

# Stage 1 — STL ball query radii
surface_radii:               [0.01, 0.05, 1.0]    # meters
surface_neighbors_in_radius: [8,    16,   128]
volume_radii:                [0.1,  0.5,  1.0, 2.5]
volume_neighbors_in_radius:  [32,   64,   128, 256]

# Stage 1 — SDF scaling factors
surface_sdf_scaling_factor:  [0.01, 0.02, 0.04]
volume_sdf_scaling_factor:   [0.04]

# Stage 2 — local geometry radii
surface_radii:               [0.05, 0.25]
surface_neighbors_in_radius: [32,   128]
volume_radii:                [0.1,  0.25]
volume_neighbors_in_radius:  [64,   128]

# Stage 3 — neighbor stencil
num_neighbors_surface: 7     # KNN mesh neighbors
num_neighbors_volume:  10    # random sphere samples
noise_intensity:       50    # sphere radius = 1/50 = 0.02m

# MLP sizes
base_neurons (GeoConvOut):      32
base_layer (LocalPointConv):    512
base_layer (AggregationModel):  512  # 4 hidden layers of 512
base_layer (FourierMLP):        512
num_modes (Fourier):            5
```

---

## What Makes DoMINO Unique

1. **Hybrid point-cloud + volumetric latent space** — raw STL points are projected onto a regular 3D grid enabling efficient CNN processing, while query points remain arbitrary and mesh-agnostic.

2. **Multi-scale ball queries at both stages** — different radii capture fine surface curvature (1cm) and long-range pressure wakes (2.5m) simultaneously, critical for elliptic PDEs.

3. **Decomposability** — Stage 1 runs once per geometry regardless of how many query points exist. Stage 2 is O(N_points) and independent of geometry complexity. This makes inference scale to millions of mesh nodes.

4. **Dual SDF + STL encoding** — SDF provides smooth implicit geometry (orientation, inside/outside); explicit STL ball queries provide exact surface geometry. Together they handle both smooth panels and sharp features.

5. **Learned finite-difference stencil** — predicting at the center point plus neighbors with inverse-distance blending forces spatial consistency without requiring automatic differentiation through the model.

6. **Per-variable independent networks** — each output field (pressure, Vx, Vy, etc.) has its own basis function and aggregation MLP, allowing the model to learn field-specific spatial representations.
