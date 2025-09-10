# Simulation of Low-Frequency Variability in the Equatorial Pacific

## Overview

This experiment aims to simulate low-frequency variability in the deep equatorial Pacific Ocean using the MITgcm (MIT General Circulation Model). The simulation is forced by surface wind stress fields derived from the ERA5 reanalysis dataset, with the goal of reproducing observed low-frequency signals in the deep equatorial ocean, such as those associated with interannual to decadal variability (e.g., related to ENSO or deeper equatorial waves).

## Objectives

- Reproduce low-frequency (interannual to decadal) oceanic signals in the deep equatorial Pacific.
- Validate model output against available observational datasets (e.g., TAO/TRITON moorings, Argo floats, or historical hydrographic sections).
- Investigate the role of wind stress forcing from ERA5 in exciting and deep equatorial variability.

## Model Configuration

- **Model**: MITgcm (checkpoint version: `checkpoint68r`)
- **Domain**: Equatorial Pacific (e.g., 20°S–20°N, 120°E–80°W)
- **Resolution**: Horizontal: 1/4° (~25 km), Vertical: 50 levels (enhanced resolution near thermocline and equator)
- **Forcing**: Surface wind stress from ERA5 (monthly or daily, 1979–present)
- **Boundary Conditions**: Restoring for temperature and salinity at open boundaries (optional: sponge layers)
- **Initial Conditions**: From WOA climatology or spun-up state
- **Integration Period**: 1980–2020 (with 5–10 year spin-up)
