# Simulation of Deep Low-Frequency Variability in the Equatorial Pacific

## Overview

This experiment aims to simulate low-frequency variability in the deep equatorial Pacific Ocean using the MITgcm (MIT General Circulation Model). The simulation is forced by surface wind stress fields derived from the ERA5 reanalysis dataset, with the goal of reproducing observed low-frequency signals in the deep equatorial ocean, such as those associated with interannual to decadal variability (e.g., related to ENSO or deeper equatorial waves).

## Objectives

- Reproduce low-frequency (interannual to decadal) oceanic signals in the deep equatorial Pacific.
- Validate model output against available observational datasets (e.g., TAO/TRITON moorings, Argo floats, or historical hydrographic sections).
- Investigate the role of wind stress forcing from ERA5 in exciting and deep equatorial variability.

## Model Configuration

- **Model**: MITgcm (checkpoint version: `checkpoint69r`)
- **Domain**: Equatorial Pacific (e.g., 25°S–25°N, 100°E–80°W)
- **Resolution**: Horizontal: 1/4° (~25 km), Vertical: 50 levels (enhanced resolution near thermocline and equator)
- **Forcing**: Surface wind stress from ERA5 (monthly or daily, 1979–present)
- **Boundary Conditions**: Restoring for temperature and salinity at open boundaries (optional: sponge layers)
- **Initial Conditions**: From WOA climatology or spun-up state
- **Integration Period**: 1980–2020 (with 5–10 year spin-up)

## Repository layout

```text
bin/                     Executable commands (Python or Bash)
scripts/preprocess/      grid/, boundaries/, forcing/ input-generation recipes
scripts/postprocess/     Export and runtime statistics
analysis/                Editable scientific analyses, quicklooks and notebooks
src/mitkit/              Shared I/O, array helpers, ocean modes and paths
archive/                 Historical recipes retained for comparison
code/                    MITgcm source customizations and compile-time options
config/                  Namelist templates
input/                   Active model inputs
data/                    Forcing and supporting datasets
build/                   Build products
output/                  Experiment run directories and model outputs
pickup/                  Restart files
fig/                     Scientific figures
log/                     Logs
```

The script migration does not move or alter model files in `code/`, `config/`,
`input/`, `data/`, `build/`, `output/` or `pickup/`.

## Python environment and commands

In the existing `py312` environment, install the shared code without changing
scientific dependencies (run from the repository root):

```bash
conda activate py312
python -m pip install --no-deps --no-build-isolation -e .
export PATH="/public/home/zhanghang/eqp-dlfv-mitgcm-era5/bin:$PATH"
```

Replace the old `tool` PATH entry with `bin` in your shell configuration if one
exists. Python commands use `#!/usr/bin/env python`, hence the active environment.
For a new environment, `python -m pip install -e '.[analysis,forcing]'` installs
the declared dependencies. MITgcm, MPI/Slurm, `rclone`, and `gluemncbig` remain
external tools where needed.

```bash
mitmon --help
mitmon -1
mitrun --help
link-pickup --help
```

`bin/mitmon` contains the Python monitoring implementation directly; no Bash
wrapper is required. `mitrun` invokes the adjacent `link-pickup` executable.
`mitglue` retains its existing `run/mnc_exp_*` input convention; it is a legacy
MNC workflow, not an automatic merger for every directory under `output/`.

## Processing and analysis

Edit parameters in each recipe before execution, then run it with Python:

```bash
python scripts/preprocess/forcing/gen_filtwind.py
python scripts/postprocess/export_nc.py
python analysis/ceof/ana_ceof.py
```

Preprocessing can write model inputs; review input/output settings before running.
Scientific scripts keep editable parameter blocks and `# %%` cells. Mixing
analyses containing `display()` should be run in IPython or VS Code's interactive
window, using the same environment where the package is installed.

Shared imports use `from mitkit.io import open_mds, parse_diag`.
Read Fortran namelists with `f90nml.read(path)`; the old `parse_file` API has
been removed. Project paths use
`mitkit.paths.project_root()`: an explicit `WORK_DIR` takes precedence;
otherwise the editable checkout is used, independent of the working directory.
External source datasets default to the existing `~/data` locations where used.

Former `script/analysis` files are under `analysis/ceof`, `mixing`, and `waves`;
`mitshow_*.py` are editable recipes in `analysis/quicklook`. The unvalidated
plotting framework and its example are archived under `archive/mitkit_review`,
outside the installed package. The hydrology notebook is retained under
`analysis/notebooks` and is no longer ignored by Git. Historical generators are
kept under `archive/preprocess` with their original contents.

The scope and findings of the shared-code review are documented in
[`archive/mitkit_review/README.md`](archive/mitkit_review/README.md).
Run the focused I/O regression checks with `python -m unittest discover -s tests`.
