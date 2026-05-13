# DoMINO Training Pipeline

End-to-end guide for converting CFD data, training a DoMINO surrogate model,
and running inference on new geometries.

---

## Pipeline Overview

```
Raw CFD data                    Zarr dataset                   Trained model
                                (one .zarr per case)
       │                               │                              │
       ▼                               ▼                              ▼
[VTI-based]          validate_zarr.py                          run_inference.py
convert_to_zarr.py   inspect_zarr.py                           (new STL → VTI)
                     check_coords.py
[CSV-based]          split_zarr.py  ──►  zarr_val/  (val cases moved out)
csv_to_zarr.py                      └── zarr/      (train, unchanged)
                                                  │
[Combined: run both                               ▼
scripts — they write          shuffle_zarr_volume.py  ──►  zarr_shuffled/
to the same .zarr]            (recommended for volume_sample_from_disk: true)
                                                  │
                                                  ▼
                                         check_bounds.py  ──►  config.yaml
                                         check_areas.py   ──►  config.yaml
                                         compute_statistics.py ──►  scaling_factors.pkl
                                                  │
                                                  ▼
                                             train.py
```

---

## Prerequisites

**Conda environment:**
```bash
conda activate domino
```

**Working directory** — all scripts must be run from:
```bash
cd /path/to/physicsnemo/examples/cfd/external_aerodynamics/domino/src
```

**Input data layout** — each simulation case must be in its own numbered folder.

*VTI-based (volume training):*
```
data/
  0/
    mesh.stl       # surface geometry
    result.vti     # CFD volume results
  1/
    mesh.stl
    result.vti
  ...
```

*CSV-based (LBM cut-cell / surface training):*
```
data/
  0/
    mesh.stl           # surface geometry
    pressure.csv       # columns: x, y, z, p
    velocity.csv       # columns: x, y, z, vx, vy, vz
  1/
    mesh.stl
    pressure.csv
    velocity.csv
  ...
```

---

## Step 1 — Convert CFD Data to Zarr

Two scripts are available depending on your data format.  Both produce `.zarr`
files with the same STL keys; the difference is in what field arrays they write.

---

### Step 1a — STL + VTI → Zarr (volume training)

**Script:** `convert_to_zarr.py`

Converts each case folder (STL + VTI) into a `.zarr` file for volume or
combined DoMINO training. Also handles unit conversion if the STL is in
millimetres.

#### Configuration

Edit the constants at the top of `convert_to_zarr.py`:

| Constant | Description |
|---|---|
| `INPUT_DIR` | Parent folder containing the numbered case folders |
| `OUTPUT_DIR` | Where `.zarr` files will be written |
| `STL_FILENAME` | STL file name inside each case folder (default: `mesh.stl`) |
| `VTI_FILENAME` | VTI file name inside each case folder (default: `result.vti`) |
| `STL_UNIT_SCALE` | `0.001` if STL is in mm, `1.0` if already in metres |
| `INLET_VELOCITY` | Inlet velocity in m/s (must match your simulation) |
| `AIR_DENSITY` | Air density in kg/m³ (default: `1.225`) |
| `VOLUME_FIELD_NAMES` | Dict mapping VTI field names → `"vector"` or `"scalar"` |
| `FIELD_LOCATION` | `"point_data"` or `"cell_data"` |
| `SKIP_EXISTING` | `True` to skip already-converted cases (safe for re-runs) |

#### Usage

**First run — discover field names (leave `VOLUME_FIELD_NAMES = {}`):**
```bash
python convert_to_zarr.py
```
The script prints all available fields from your VTI with a ready-to-paste
`VOLUME_FIELD_NAMES` template, then exits.

**Second run — convert all cases:**

Fill in `VOLUME_FIELD_NAMES` with the names printed above, then run again:
```bash
python convert_to_zarr.py
```

If a field name is wrong the script errors immediately and shows the correct names.

#### Expected output
```
Found 50 cases in /data/raw
[0]
  OK: stl=12450 verts / 24896 faces | vol=524288 pts / 4 channels
[1]
  OK: stl=11980 verts / 23956 faces | vol=524288 pts / 4 channels
...
Done: 50 converted, 0 failed
```

> **Note:** The column order in `VOLUME_FIELD_NAMES` determines the column order
> in `volume_fields` and **must match** the `variables.volume.solution` section
> in `config.yaml`.

---

### Step 1b — STL + CSV → Zarr (LBM cut-cell / surface training)

**Script:** `csv_to_zarr.py`

Converts each case folder (STL + two CSV files) into a `.zarr` file for surface
or combined DoMINO training. The CSVs contain LBM cut-cell centers (voxels
intersected by the STL surface). Surface normals and areas are derived by
projecting each cut-cell center onto the nearest STL face.

Zarr stores are opened with `mode="a"` (append), so you can run
`convert_to_zarr.py` first and then `csv_to_zarr.py` to produce a single zarr
with both volume and surface keys for `model_type: combined`.

#### Configuration

Edit the constants at the top of `csv_to_zarr.py`:

| Constant | Description |
|---|---|
| `INPUT_DIR` | Parent folder containing the numbered case folders |
| `OUTPUT_DIR` | Where `.zarr` files will be written |
| `STL_FILENAME` | STL file name (default: `mesh.stl`) |
| `PRESSURE_CSV_FILENAME` | Pressure CSV file name (default: `pressure.csv`) |
| `VELOCITY_CSV_FILENAME` | Velocity CSV file name (default: `velocity.csv`) |
| `COORD_COLS` | 0-based column indices for x, y, z in both CSVs (default: `[0, 1, 2]`) |
| `PRESSURE_COLS` | Column indices for pressure values (default: `[3]`) |
| `VELOCITY_COLS` | Column indices for velocity values (default: `[3, 4, 5]`) |
| `CSV_HAS_HEADER` | `True` if CSVs have a header row, `False` if purely numeric |
| `STL_UNIT_SCALE` | `0.001` if STL/CSV coordinates are in mm, `1.0` if in metres |
| `INLET_VELOCITY` | Inlet velocity in m/s |
| `AIR_DENSITY` | Air density in kg/m³ (default: `1.225`) |
| `SKIP_EXISTING` | `True` to skip already-converted cases |

#### Zarr keys written

| Key | Shape | Description |
|---|---|---|
| `stl_coordinates` | `[N_verts, 3]` | STL vertex positions |
| `stl_centers` | `[N_faces, 3]` | STL face centres |
| `stl_faces` | `[N_faces*3]` | Triangle vertex indices |
| `stl_areas` | `[N_faces]` | Triangle areas (m²) |
| `surface_mesh_centers` | `[N, 3]` | Cut-cell centre coordinates |
| `surface_normals` | `[N, 3]` | Outward face normals (from nearest STL face) |
| `surface_areas` | `[N]` | Face areas inherited from nearest STL face |
| `surface_fields` | `[N, 4]` | Fields in column order `[p, vx, vy, vz]` |
| `global_params_values` | `[2, 1]` | `[[inlet_velocity], [air_density]]` |
| `global_params_reference` | `[2, 1]` | Same as above |

#### Usage

```bash
python csv_to_zarr.py
```

The script runs a one-time CSV inspection (row/column counts, first 3 rows)
before converting all cases.

#### Expected output
```
Found 50 cases in /data/raw

── CSV inspection (case: 0) ──
  pressure.csv: 82341 rows, 4 columns
    first 3 rows:
    ...
  velocity.csv: 82341 rows, 6 columns
    ...

[0]
  OK: stl=12450 verts / 24896 faces | surface=82341 pts / 4 channels
[1]
  OK: stl=11980 verts / 23956 faces | surface=80122 pts / 4 channels
...
Done: 50 converted, 0 failed
```

> **Note:** The column order of `surface_fields` is `[p, vx, vy, vz]` and must
> match the `variables.surface.solution` section in `config.yaml`.

---

## Step 2 — Validate Zarr Files

**Script:** `validate_zarr.py`

Checks every `.zarr` case for correct keys, shapes, dtypes, NaN/Inf values,
and geometry consistency. Run this before doing anything else with your data.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--data_dir` | yes | Directory containing `.zarr` case folders |
| `--verbose` / `-v` | no | Print per-case stats (velocity range, pressure range, point counts) |

### Usage
```bash
python validate_zarr.py --data_dir /data/zarr
python validate_zarr.py --data_dir /data/zarr --verbose
```

### Expected output
```
Validating 50 cases in /data/zarr

  PASS  0.zarr
  PASS  1.zarr
  FAIL  2.zarr
        ! volume_fields contains 12 NaN values

Result: 49 passed, 1 failed
```

Fix any failed cases before continuing.

---

## Step 3 — Inspect Field Distributions

**Script:** `inspect_zarr.py`

Plots histograms and box plots of every field across all cases. Useful for
spotting unit errors, outliers, and understanding the data ranges before training.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--data_dir` | yes | Directory containing `.zarr` folders or `.npy` files |
| `--fields` | no | Only plot fields whose names contain these strings |
| `--max_cases` | no | Limit to this many cases (faster for large datasets) |
| `--save_dir` | no | Save plots as PNG files here instead of showing interactively |

### Usage
```bash
# Plot all fields interactively:
python inspect_zarr.py --data_dir /data/zarr

# Only check velocity and pressure, save to disk:
python inspect_zarr.py --data_dir /data/zarr --fields volume_fields stl_areas --save_dir /data/plots

# Quick check on 5 cases:
python inspect_zarr.py --data_dir /data/zarr --max_cases 5
```

---

## Step 4 — Check Coordinate Alignment

**Script:** `check_coords.py`

Compares the spatial extents of your training zarr data, the config bounding
boxes, and an optional inference STL side by side. Run this if you suspect a
coordinate system or unit mismatch.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--zarr_dir` | yes | Training zarr directory |
| `--config` | yes | Path to `config.yaml` |
| `--stl` | no | Path to an inference STL to compare |
| `--n_cases` | no | Number of zarr cases to sample |

### Usage
```bash
python check_coords.py --zarr_dir /data/zarr_train --config conf/config.yaml
python check_coords.py --zarr_dir /data/zarr_train --config conf/config.yaml --stl /data/new/mesh.stl
```

---

## Step 5 — Split Train / Val

**Script:** `split_zarr.py`

Moves validation cases out of `zarr_dir` into a separate `zarr_val/` directory.
`zarr_dir` stays in place and becomes the training directory — only the val
cases (~20%) are moved, not the full dataset.

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--zarr_dir` | yes | — | Directory with all `.zarr` cases (becomes train dir) |
| `--val_dir` | no | `<zarr_dir>/../zarr_val` | Output val directory |
| `--val_pct` | no | `0.2` | Fraction of cases for validation (e.g. `0.2` = 20%) |
| `--random` | no | off | Shuffle cases randomly before splitting |
| `--seed` | no | `42` | Random seed for reproducibility (used with `--random`) |

### Usage
```bash
# Default: 20% val, deterministic (sorted order):
python split_zarr.py --zarr_dir /data/zarr

# Random 20% val split, reproducible:
python split_zarr.py --zarr_dir /data/zarr --val_pct 0.2 --random --seed 42

# Custom val directory:
python split_zarr.py --zarr_dir /data/zarr --val_dir /data/zarr_val
```

The script prints the exact `input_dir` / `input_dir_val` paths to set in `config.yaml`.

---

## Step 6 — Shuffle Volume Data (Recommended)

**Script:** `shuffle_zarr_volume.py`

Copies the training zarr directory to a new location with `volume_mesh_centers`
and `volume_fields` randomly permuted. This is required when using
`volume_sample_from_disk: true` with a bounding box that covers less than the
full domain.

### Why this is needed

`volume_sample_from_disk: true` reads **contiguous chunks** from zarr rather
than loading all points. It assumes the data is pre-shuffled so that contiguous
chunks are spatially random. CFD data is stored in mesh order (spatially
correlated), so without shuffling, a chunk may land entirely outside the
bounding box and cause:

```
ValueError: Volume mesh has fewer points than requested sample size
```

Only the training dataset needs shuffling. Validation data is not affected.

### What is shuffled

Both arrays are permuted with the **same random index** to keep them aligned:

| Array | Shape | Action |
|---|---|---|
| `volume_mesh_centers` | [N, 3] | Shuffled |
| `volume_fields` | [N, C] | Shuffled (same permutation) |
| `stl_coordinates`, `stl_centers`, `stl_faces`, `stl_areas` | — | Copied as-is |
| `global_params_values`, `global_params_reference` | — | Copied as-is |

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--src_dir` | yes | — | Source zarr directory (original training data) |
| `--dst_dir` | yes | — | Destination for shuffled zarr cases |
| `--seed` | no | `42` | Random seed for reproducibility |
| `--dry_run` | no | off | Show what would be done without writing files |
| `--skip_done` | no | off | Skip cases already present in `dst_dir` (for resuming) |

### Usage
```bash
# Dry run first:
python shuffle_zarr_volume.py \
    --src_dir /data/zarr_data \
    --dst_dir /data/zarr_data_shuffled \
    --dry_run

# Shuffle all training cases:
python shuffle_zarr_volume.py \
    --src_dir /data/zarr_data \
    --dst_dir /data/zarr_data_shuffled

# Resume an interrupted run:
python shuffle_zarr_volume.py \
    --src_dir /data/zarr_data \
    --dst_dir /data/zarr_data_shuffled \
    --skip_done
```

After running, update `config.yaml`:
```yaml
data:
  input_dir: /data/zarr_data_shuffled
  volume_sample_from_disk: true
```

> **Note:** Each case loads ~1.1 GB into RAM during shuffling (40M points × 7
> fields × 4 bytes). Expect ~1–2 minutes per case. The original `zarr_data`
> is not modified.

---

## Step 7 — Get Bounding Boxes

**Script:** `check_bounds.py`

Reports the aggregate bounding box of STL geometry and volume mesh across all
cases, with a ready-to-paste `config.yaml` snippet.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--data_dir` | yes | Directory containing `.zarr` case folders |
| `--max_cases` | no | Limit to this many cases (faster for large datasets) |

### Usage
```bash
python check_bounds.py --data_dir /data/zarr
```

### Example output
```
STL surface bounds  (bounding_box_surface):
  config.yaml snippet:
    min: [-0.04, -1.59, -0.13]
    max: [2.94,  1.59,  0.43]

Volume mesh bounds  (bounding_box):
  config.yaml snippet:
    min: [-2.0, -3.7, -2.7]
    max: [7.2,  3.5,  2.3]
```

Paste these values into `config.yaml` under `data.bounding_box` and
`data.bounding_box_surface`.

---

## Step 8 — Get Area Weighing Factor

**Script:** `check_areas.py`

Reports STL face area statistics and suggests an `area_weighing_factor` value
(≈ 1 / max_area) for `config.yaml`.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--data_dir` | yes | Directory containing `.zarr` case folders |
| `--max_cases` | no | Limit to this many cases |

### Usage
```bash
python check_areas.py --data_dir /data/zarr
```

### Example output
```
Across 40 cases:
  max area:  1.84e-02 m²
  mean area: 3.21e-04 m²

  config.yaml snippet:
    model:
      loss_function:
        area_weighing_factor: 54
```

Paste the suggested value into `config.yaml`.

---

## Step 9 — Update config.yaml

Edit `conf/config.yaml` with the values from the previous steps:

**Volume training** (`convert_to_zarr.py` data):
```yaml
project:
  name: RAF_CFD

exp_tag: 1   # increment for each new training run

output: /path/to/outputs/${project.name}/${exp_tag}   # where checkpoints/logs are saved

data:
  input_dir: /path/to/zarr                # original zarr_dir (train cases remain here)
  input_dir_val: /path/to/zarr_val        # from split_zarr.py
  scaling_factors: /path/to/scaling_factors/scaling_factors.pkl
  max_samples_for_statistics: 200         # set to total number of training cases
  bounding_box:                           # from check_bounds.py
    min: [-2.0, -3.7, -2.7]
    max: [7.2,  3.5,  2.3]
  bounding_box_surface:                   # from check_bounds.py
    min: [-0.04, -1.59, -0.13]
    max: [2.94,  1.59,  0.43]

model:
  model_type: volume                      # volume / surface / combined
  loss_function:
    area_weighing_factor: 54              # from check_areas.py

variables:
  volume:
    solution:
      # Column order must match VOLUME_FIELD_NAMES in convert_to_zarr.py
      U_time_avg: vector   # columns 0, 1, 2
      p_time_avg: scalar   # column 3
  surface:
    solution: {}           # empty — not used for volume-only training
```

**Surface training** (`csv_to_zarr.py` data):
```yaml
model:
  model_type: surface                     # volume / surface / combined
  use_surface_normals: true
  use_surface_area: true
  surface_sampling_algorithm: area_weighted
  loss_function:
    area_weighing_factor: 54              # from check_areas.py

variables:
  surface:
    solution:
      # Column order must match surface_fields written by csv_to_zarr.py: [p, vx, vy, vz]
      pressure: scalar     # column 0
      velocity: vector     # columns 1, 2, 3
  volume:
    solution: {}           # empty — not used for surface-only training
```

> **Notes:**
> - `output` controls where checkpoints, TensorBoard logs, and Hydra config are saved. Set to an absolute path (e.g. on a mounted GCS bucket) to persist across sessions.
> - `data_processor`, `project_dir`, and `train.checkpoint_dir` are legacy fields — leave them as-is, they are not used by `train.py`.
> - Variable names and order under `variables.volume.solution` / `variables.surface.solution` must exactly match the column order written by `convert_to_zarr.py` / `csv_to_zarr.py` respectively.

---

## Step 10 — Compute Scaling Factors

**Script:** `compute_statistics.py`

Computes mean, std, min, and max across the training dataset and saves them to
`scaling_factors.pkl`. This file is required by both training and inference.

Makes **two passes** over the dataset: first to compute mean/std, then to
compute min/max with outlier filtering (±9σ). Expect roughly 2× the time of a
single read pass.

### Prerequisites
`config.yaml` must have the following set before running:
- `model.model_type` — determines which fields to compute stats for
- `data.bounding_box` and `data.bounding_box_surface` — from `check_bounds.py`

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_dir` | yes | — | Training zarr directory |
| `--output` | yes | — | Full path to save `scaling_factors.pkl` (must end in `.pkl`) |
| `--config` | no | `conf/config.yaml` | Path to config file |
| `--max_samples` | no | from config | Max number of cases to sample (default 200) |
| `--force` | no | off | Recompute even if `.pkl` already exists |

### Usage
```bash
python compute_statistics.py \
    --data_dir /data/zarr \
    --output /data/scaling_factors/scaling_factors.pkl
```

### Output
- `scaling_factors.pkl` — loaded by training and inference
- `scaling_factors_summary.txt` — human-readable report of all statistics

> **Note:** `--output` must be a full file path ending in `.pkl`, not just a
> directory. Make sure `data.scaling_factors` in `config.yaml` points to the
> same path.

---

## Step 11 — Train

**Script:** `train.py`

Reads all settings from `conf/config.yaml` (Hydra). Outputs checkpoints,
logs, and TensorBoard events to `outputs/<project.name>/<exp_tag>/`.

### Usage
```bash
# Single GPU:
python train.py

# Multi-GPU (recommended):
torchrun --nproc_per_node=8 train.py

# Override specific config values without editing the file:
torchrun --nproc_per_node=8 train.py exp_tag=2 train.epochs=500
```

### Prerequisites
```bash
pip install pynvml
```

### Key config parameters

| Parameter | Description |
|---|---|
| `exp_tag` | Experiment number — increment for each new run |
| `train.epochs` | Number of training epochs |
| `train.optimizer.lr` | Learning rate |
| `train.lr_scheduler.name` | `MultiStepLR` or `CosineAnnealingLR` |
| `model.model_type` | `volume`, `surface`, or `combined` |
| `model.volume_points_sample` | Points sampled per epoch during training |
| `model.normalization` | `min_max_scaling` or `mean_std_scaling` |

### Output directory structure
```
outputs/RAF_CFD/2/
  models/          # checkpoints (DoMINO.0.{epoch}.mdlus)
  tensorboard/     # TensorBoard event files
  hydra/           # saved config for reproducibility
  train.log        # training log
```

---

## Step 12 — Monitor Training with TensorBoard

TensorBoard events are written to `outputs/<project.name>/<exp_tag>/tensorboard/`
during training. The following metrics are logged:

- `Loss/train` — training loss per epoch
- `L2 Metrics/train/<field>` — per-field L2 error on training set
- `L2 Metrics/val/<field>` — per-field L2 error on validation set

### Launch TensorBoard
```bash
tensorboard --logdir outputs/RAF_CFD/2/tensorboard
```

Then open `http://localhost:6006` in your browser.

**To compare multiple experiments:**
```bash
tensorboard --logdir outputs/RAF_CFD
```
TensorBoard will show all `exp_tag` runs side by side.

**From WSL, access in Windows browser** at `http://localhost:6006` — the port
is forwarded automatically.

---

## Step 13 — Inference

**Script:** `run_inference.py`

Loads a trained checkpoint and predicts `[Vx, Vy, Vz, P]` on a new STL
geometry. Outputs a VTI file (grid-based) or VTU file (scattered points).

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--stl` | yes | — | Path to input STL file |
| `--checkpoint` | yes | — | Directory containing `.mdlus` checkpoint files |
| `--output` | no | `predicted_volume.vtu` | Output file path (`.vti` for grid, `.vtu` for points) |
| `--stl_scale` | no | `1.0` | Scale STL coordinates (use `0.001` for mm → m) |
| `--vti_resolution` | no | from config | Grid resolution `NX NY NZ` (only for `.vti` output) |
| `--num_points` | no | `500000` | Number of points to predict (only for `.vtu` output) |
| `--inlet_velocity` | no | from config | Override inlet velocity (m/s) |
| `--air_density` | no | from config | Override air density (kg/m³) |
| `--config` | no | auto-detected | Path to `config.yaml` |
| `--scaling` | no | from config | Path to `scaling_factors.pkl` |
| `--batch_size` | no | from config | Points per inference batch |

### Usage

**VTI output (grid-based, recommended for ParaView):**
```bash
python run_inference.py \
    --stl        /data/new/mesh.stl \
    --checkpoint outputs/RAF_CFD/2/models \
    --output     /data/predicted.vti \
    --stl_scale  0.001 \
    --vti_resolution 128 64 64
```

**VTU output (scattered points):**
```bash
python run_inference.py \
    --stl        /data/new/mesh.stl \
    --checkpoint outputs/RAF_CFD/2/models \
    --output     /data/predicted.vtu \
    --stl_scale  0.001 \
    --num_points 500000
```

### Output VTI arrays

| Array | Type | Description |
|---|---|---|
| `velocity_time_avg` | vector (3-component) | Time-averaged velocity [Vx, Vy, Vz] |
| `pressure_time_avg` | scalar | Time-averaged pressure |
| `ImplicitField` | scalar | Fluid domain mask (−1 everywhere) |

> **Note:** The script auto-detects the latest checkpoint in the `--checkpoint`
> directory. The config and scaling factors are read from the checkpoint
> directory automatically (saved there by `train.py`), so `--config` and
> `--scaling` overrides are usually not needed.

### Programmatic usage — multiple geometries

For servers or batch workflows that process many geometries per session, use
`DoMINORunner` directly.  Config load, scaling factors, datapipe construction,
and model-weight deserialisation (the 6–18 s fixed overhead) happen **once** in
the constructor.  Each `infer()` call only performs the per-geometry work (STL
load + model forward).

```python
from run_inference import DoMINORunner

# --- startup: expensive, runs once ---
runner = DoMINORunner(
    checkpoint_dir="outputs/RAF_CFD/1/models",
    # config_path and scaling_path are auto-detected from the checkpoint dir
)

# --- per-geometry: fast ---
runner.infer("car_v1.stl",  "car_v1.vtu")
runner.infer("car_v2.stl",  "car_v2.vtu")
runner.infer("truck.stl",   "truck.vti",  inlet_velocity=30.0)
runner.infer("wing.stl",    "wing.vtu",   stl_scale=0.001, num_points=1_000_000)
```

**`DoMINORunner` constructor parameters:**

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_dir` | required | Directory containing `.mdlus` files |
| `config_path` | auto-detected | Path to `config.yaml` |
| `scaling_path` | from config | Path to `scaling_factors.pkl` |
| `device` | GPU 0 | `torch.device` or string |

**`runner.infer()` parameters** (all keyword-only except the first two):

| Parameter | Default | Description |
|---|---|---|
| `stl_path` | required | Input STL file |
| `output_path` | required | Output file (`.vti` or `.vtu`) |
| `inlet_velocity` | from config | Override inlet velocity (m/s) |
| `air_density` | from config | Override air density (kg/m³) |
| `num_points` | `500_000` | Points to predict (VTU mode) |
| `batch_size` | from config | Points per inference batch |
| `stl_scale` | `1.0` | Coordinate scale factor (use `0.001` for mm → m) |
| `vti_resolution` | from config | `(nx, ny, nz)` for VTI output |

The CLI (`python run_inference.py ...`) is unchanged and internally instantiates
`DoMINORunner` for a single geometry.

---

## Troubleshooting

**`No exterior cells found` during inference**
> The STL geometry does not overlap with the bounding box in `config.yaml`, or
> the STL is in the wrong units. Run `check_coords.py` to diagnose, and use
> `--stl_scale 0.001` if the STL is in millimetres.

**`fields not found in point_data` during conversion**
> The field names in `VOLUME_FIELD_NAMES` don't match the actual VTI field names.
> Set `VOLUME_FIELD_NAMES = {}` and re-run to get the correct names.

**`Row count mismatch` or `Coordinate mismatch` during CSV conversion**
> `pressure.csv` and `velocity.csv` must have the same number of rows and the
> same x, y, z coordinates. Verify that both files come from the same simulation
> output. The script asserts coordinate agreement to within 1e-6 before stacking fields.

**`IndexError` or wrong values after CSV conversion**
> Check `COORD_COLS`, `PRESSURE_COLS`, and `VELOCITY_COLS` in `csv_to_zarr.py`
> against the actual column layout of your CSVs. The one-time inspection printed
> at startup shows the first 3 rows — use those to verify the indices.
> If your CSVs have a header row, set `CSV_HAS_HEADER = True`.

**`Missing key` errors in `validate_zarr.py`**
> A required array is missing from a zarr case — the conversion likely failed or
> was incomplete for that case. Re-run `convert_to_zarr.py` with `SKIP_EXISTING = False`.

**`scaling_factors.pkl` not found during training**
> Run `compute_statistics.py` first, and make sure `data.scaling_factors` in
> `config.yaml` points to the correct path.

**`ValueError: Volume mesh has fewer points than requested sample size`**
> Occurs when `volume_sample_from_disk: true` reads a contiguous chunk that lands
> outside the bounding box. The zarr volume data must be shuffled first.
> Run `shuffle_zarr_volume.py` on the training data, then point `input_dir` to
> the shuffled directory. Alternatively set `volume_sample_from_disk: false` to
> load all points (more IO, but always correct).

**Training loss not decreasing**
> - Check `area_weighing_factor` — run `check_areas.py` to verify the value
> - Check bounding boxes — run `check_bounds.py` to verify `bounding_box` covers the full domain
> - Inspect data quality — run `inspect_zarr.py` to check for outliers or unit issues
