from mitkit.paths import project_root
from pathlib import Path
import calendar
import os

import numpy as np

# ============================================================
# Settings
# ============================================================

BASE_DIR = project_root()
EXF_DIR = BASE_DIR / "data/exf"

PATH_IN = EXF_DIR / "era5_dy"
PATH_OUT = EXF_DIR / "wind_mean"

YEARS = range(1990, 2026)

NX = 761
NY = 241
DTYPE = ">f4"                      # MITgcm big-endian float32

# Only variables listed here are processed.
VARIABLES = {
    "uwind": ("uwind_{year}", "uwind.bin"),
    "vwind": ("vwind_{year}", "vwind.bin"),
}

# ============================================================
# File information
# ============================================================

YEARS = list(YEARS)
bytes_per_day = NY * NX * np.dtype(DTYPE).itemsize


def get_ndays(path):
    """Return the number of complete daily records in a binary file."""
    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    if size % bytes_per_day != 0:
        raise ValueError(
            f"{path}\n"
            "File size is not an integer number of daily records.\n"
            f"Actual: {size} bytes\n"
            f"Bytes per day: {bytes_per_day}"
        )

    ndays = size // bytes_per_day
    if ndays == 0:
        raise ValueError(f"Empty input file: {path}")

    return ndays


# ============================================================
# Process one forcing variable
# ============================================================

def process_variable(name, template, output_filename):
    print(f"\nProcessing {name}")

    field_sum = np.zeros((NY, NX), dtype=np.float64)
    total_days = 0

    for year in YEARS:
        path = PATH_IN / template.format(year=year)
        ndays = get_ndays(path)
        expected_days = 366 if calendar.isleap(year) else 365

        if ndays != expected_days:
            print(
                f"{name} {year}: excluded incomplete year "
                f"({ndays}/{expected_days} days)"
            )
            continue

        src = np.fromfile(path, dtype=DTYPE).reshape(ndays, NY, NX)
        field_sum += np.sum(src, axis=0, dtype=np.float64)

        total_days += ndays
        print(f"{name} {year}: accumulated {ndays} days")

    if total_days == 0:
        raise ValueError(f"No complete years found for {name}")

    mean_field = (field_sum / total_days).astype(DTYPE)
    mean_field.tofile(PATH_OUT / output_filename)

    print(
        f"{name}: mean of {total_days} daily records written to "
        f"{PATH_OUT / output_filename}"
    )


# ============================================================
# Run
# ============================================================

if PATH_OUT == PATH_IN:
    raise ValueError("PATH_OUT must differ from PATH_IN")

if not VARIABLES:
    raise ValueError("VARIABLES must contain at least one variable")

PATH_OUT.mkdir(parents=True, exist_ok=True)

print(f"Input:  {PATH_IN}")
print(f"Output: {PATH_OUT}")
print(f"Years:  {YEARS[0]}-{YEARS[-1]}")
print(f"Variables: {', '.join(VARIABLES)}")

for variable_name, file_names in VARIABLES.items():
    input_template, output_filename = file_names
    process_variable(variable_name, input_template, output_filename)

print("\nAll done.")
