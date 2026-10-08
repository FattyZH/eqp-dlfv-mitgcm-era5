from mitkit.paths import project_root
from pathlib import Path
from numbers import Real
import os

import numpy as np

# ============================================================
# Settings
# ============================================================

BASE_DIR = project_root()
EXF_DIR = BASE_DIR / "data/exf"
INPUT_DIR = EXF_DIR

# (directory name, "yearly" or "constant", coefficient)
INPUT_FIELDS = [
    ("era5_dy", "yearly", 1),
    ("wind_bp24-96", "yearly", -1),
]

PATH_OUT = EXF_DIR / "wind_bs24-96"

YEARS = range(1990, 2027)

NX = 761
NY = 241
DTYPE = ">f4"                      # MITgcm big-endian float32

# Only variables listed here are processed.
VARIABLES = {
    "uwind": ("uwind_{year}", "uwind.bin"),
    "vwind": ("vwind_{year}", "vwind.bin"),
}

# ============================================================
# Checks
# ============================================================

if not INPUT_FIELDS:
    raise ValueError("INPUT_FIELDS must contain at least one field")

for index, field in enumerate(INPUT_FIELDS, start=1):
    if len(field) != 3:
        raise ValueError(
            f"INPUT_FIELDS[{index}] must be (directory, type, coefficient)"
        )

    directory, field_type, coefficient = field

    if field_type not in {"yearly", "constant"}:
        raise ValueError(
            f"INPUT_FIELDS[{index}]: type must be 'yearly' or 'constant'"
        )
    if not isinstance(coefficient, Real) or not np.isfinite(coefficient):
        raise ValueError(
            f"INPUT_FIELDS[{index}]: coefficient must be a finite number"
        )
    if PATH_OUT == INPUT_DIR / directory:
        raise ValueError("PATH_OUT must differ from every input directory")

if not VARIABLES:
    raise ValueError("VARIABLES must contain at least one variable")

HAS_YEARLY_FIELDS = any(field_type == "yearly" for _, field_type, _ in INPUT_FIELDS)
bytes_per_record = NY * NX * np.dtype(DTYPE).itemsize


def get_nrecords(path):
    """Return the number of complete 2-D records in a binary file."""
    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    if size % bytes_per_record != 0:
        raise ValueError(
            f"{path}\n"
            "File size is not an integer number of 2-D records.\n"
            f"Actual: {size} bytes\n"
            f"Bytes per record: {bytes_per_record}"
        )

    nrecords = size // bytes_per_record
    if nrecords == 0:
        raise ValueError(f"Empty input file: {path}")

    return nrecords


# ============================================================
# Process one variable file
# ============================================================

def process_file(name, yearly_template, constant_filename, year=None):
    yearly_filename = (
        yearly_template.format(year=year) if year is not None else None
    )
    output_filename = yearly_filename or constant_filename

    source_files = []
    yearly_nrecords = set()

    for directory, field_type, coefficient in INPUT_FIELDS:
        filename = (
            constant_filename
            if field_type == "constant"
            else yearly_filename
        )
        path = INPUT_DIR / directory / filename
        nrecords = get_nrecords(path)

        if field_type == "constant":
            if nrecords != 1:
                raise ValueError(
                    f"Constant input must contain one record: {path}"
                )
        else:
            yearly_nrecords.add(nrecords)

        source_files.append((coefficient, path, nrecords))

    if len(yearly_nrecords) > 1:
        counts = ", ".join(str(value) for value in sorted(yearly_nrecords))
        raise ValueError(
            f"Yearly record counts differ for {name} in {year}: {counts}"
        )

    nrecords_out = next(iter(yearly_nrecords), 1)
    output_shape = (nrecords_out, NY, NX)
    outfile = PATH_OUT / output_filename
    result = np.zeros(output_shape, dtype=np.float32)

    for coefficient, path, nrecords in source_files:
        values = np.fromfile(path, dtype=DTYPE).reshape(nrecords, NY, NX)
        result += coefficient * values
        del values

    result.astype(DTYPE).tofile(outfile)

    label = year if year is not None else "constant"
    print(f"{name} {label}: {nrecords_out} records")


# ============================================================
# Run
# ============================================================

PATH_OUT.mkdir(parents=True, exist_ok=True)

print("Input fields:")
for directory, field_type, coefficient in INPUT_FIELDS:
    print(
        f"  {coefficient:+g} * "
        f"{INPUT_DIR / directory} ({field_type})"
    )
print(f"Output: {PATH_OUT}")
print(f"Variables: {', '.join(VARIABLES)}")

for variable_name, file_names in VARIABLES.items():
    filename_template, constant_filename = file_names

    if HAS_YEARLY_FIELDS:
        for current_year in YEARS:
            process_file(
                variable_name,
                filename_template,
                constant_filename,
                current_year,
            )
    else:
        process_file(variable_name, filename_template, constant_filename)

print("All done.")
