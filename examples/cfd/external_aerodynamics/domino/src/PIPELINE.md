# DoMINO Training Pipeline

End-to-end guide for converting CFD data, training a DoMINO surrogate model,
and running inference on new geometries.

---

## Pipeline Overview

```
Raw CFD data                    Zarr dataset                   Trained model
(STL + VTI files)               (one .zarr per case)
       │                               │                              │
       ▼                               ▼                              ▼
convert_to_zarr.py  ──►  validate_zarr.py                      run_inference.py
                         inspect_zarr.py                        (new STL → VTI)
                         check_coords.py
                         split_zarr.py  ──►  zarr_train/
                                             zarr_val/
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

**Input data layout** — each simulation case must be in its own numbered folder:
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

---

## Step 1 — Convert STL + VTI to Zarr

**Script:** `convert_to_zarr.py`

Converts each case folder (STL + VTI) into a `.zarr` file that DoMINO can read
during training. Also handles unit conversion if the STL is in millimetres.

### Configuration

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

### Usage

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

### Expected output
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

Moves `.zarr` cases from a single directory into separate `zarr_train/` and
`zarr_val/` directories.

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--zarr_dir` | yes | — | Source directory with all `.zarr` cases |
| `--train_dir` | no | `<zarr_dir>/../zarr_train` | Output train directory |
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

# Custom output directories:
python split_zarr.py --zarr_dir /data/zarr --train_dir /data/train --val_dir /data/val
```

---

## Step 6 — Get Bounding Boxes

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
python check_bounds.py --data_dir /data/zarr_train
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

## Step 7 — Get Area Weighing Factor

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
python check_areas.py --data_dir /data/zarr_train
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

## Step 8 — Update config.yaml

Edit `conf/config.yaml` with the values from the previous steps:

```yaml
data:
  input_dir: /path/to/zarr_train          # from split_zarr.py
  input_dir_val: /path/to/zarr_val        # from split_zarr.py
  scaling_factors: /path/to/scaling_factors/scaling_factors.pkl
  bounding_box:                           # from check_bounds.py
    min: [-2.0, -3.7, -2.7]
    max: [7.2,  3.5,  2.3]
  bounding_box_surface:                   # from check_bounds.py
    min: [-0.04, -1.59, -0.13]
    max: [2.94,  1.59,  0.43]

model:
  loss_function:
    area_weighing_factor: 54              # from check_areas.py

variables:
  volume:
    solution:
      # Column order must match VOLUME_FIELD_NAMES in convert_to_zarr.py
      U_time_avg: vector   # columns 0, 1, 2
      p_time_avg: scalar   # column 3

exp_tag: 1   # increment this for each new training run
```

> **Important:** The variable names and order under `variables.volume.solution`
> must exactly match the column order used in `convert_to_zarr.py`.

---

## Step 9 — Compute Scaling Factors

**Script:** `compute_statistics.py`

Computes mean, std, min, and max across the training dataset and saves them to
`scaling_factors.pkl`. This file is required by both training and inference.

### CLI arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_dir` | yes | — | Training zarr directory |
| `--output` | yes | — | Path to save `scaling_factors.pkl` |
| `--config` | no | `conf/config.yaml` | Path to config file |
| `--max_samples` | no | from config | Max data points to sample (lower = faster, less accurate) |
| `--force` | no | off | Recompute even if `.pkl` already exists |

### Usage
```bash
python compute_statistics.py \
    --data_dir /data/zarr_train \
    --output /data/scaling_factors/scaling_factors.pkl
```

### Output
- `scaling_factors.pkl` — loaded by training and inference
- `scaling_factors_summary.txt` — human-readable report of all statistics

> **Note:** Make sure `data.scaling_factors` in `config.yaml` points to the
> same path as `--output`.

---

## Step 10 — Train

**Script:** `train.py`

Reads all settings from `conf/config.yaml` (Hydra). Outputs checkpoints,
logs, and TensorBoard events to `outputs/<project.name>/<exp_tag>/`.

### Usage
```bash
# Standard run (reads conf/config.yaml):
python train.py

# Override specific config values without editing the file:
python train.py exp_tag=2 train.epochs=500
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

## Step 11 — Monitor Training with TensorBoard

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

## Step 12 — Inference

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

---

## Troubleshooting

**`No exterior cells found` during inference**
> The STL geometry does not overlap with the bounding box in `config.yaml`, or
> the STL is in the wrong units. Run `check_coords.py` to diagnose, and use
> `--stl_scale 0.001` if the STL is in millimetres.

**`fields not found in point_data` during conversion**
> The field names in `VOLUME_FIELD_NAMES` don't match the actual VTI field names.
> Set `VOLUME_FIELD_NAMES = {}` and re-run to get the correct names.

**`Missing key` errors in `validate_zarr.py`**
> A required array is missing from a zarr case — the conversion likely failed or
> was incomplete for that case. Re-run `convert_to_zarr.py` with `SKIP_EXISTING = False`.

**`scaling_factors.pkl` not found during training**
> Run `compute_statistics.py` first, and make sure `data.scaling_factors` in
> `config.yaml` points to the correct path.

**Training loss not decreasing**
> - Check `area_weighing_factor` — run `check_areas.py` to verify the value
> - Check bounding boxes — run `check_bounds.py` to verify `bounding_box` covers the full domain
> - Inspect data quality — run `inspect_zarr.py` to check for outliers or unit issues
