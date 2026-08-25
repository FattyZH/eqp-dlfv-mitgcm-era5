from pathlib import Path
import os

import numpy as np

# ============================================================
# Settings
# ============================================================

BASE_DIR = Path(os.environ["WORK_DIR"])
EXF_DIR = BASE_DIR / "data/exf"

PATH_IN1 = EXF_DIR / "era5_dy"
PATH_IN2 = EXF_DIR / "wind_bp14-22"
PATH_OUT = EXF_DIR / "wind_bs14-22"

OPERATION = "sub"                  # "add": field1 + field2
                                   # "sub": field1 - field2

YEARS = range(1990, 2027)

NX = 761
NY = 241
DTYPE = ">f4"                      # MITgcm big-endian float32
CHUNK_Y = 32

# Only variables listed here are processed.
VARIABLES = {
    "uwind": "uwind_{year}",
    "vwind": "vwind_{year}",
}

# ============================================================
# Checks
# ============================================================

OPERATIONS = {
    "add": np.add,
    "sub": np.subtract,
}

if OPERATION not in OPERATIONS:
    raise ValueError(
        f"Invalid OPERATION: {OPERATION!r}; choose 'add' or 'sub'"
    )

if PATH_OUT in {PATH_IN1, PATH_IN2}:
    raise ValueError("PATH_OUT must differ from both input directories")

if not VARIABLES:
    raise ValueError("VARIABLES must contain at least one variable")

itemsize = np.dtype(DTYPE).itemsize
bytes_per_day = NY * NX * itemsize
operation = OPERATIONS[OPERATION]


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
# Process one variable and year
# ============================================================

def process_file(name, template, year):
    filename = template.format(year=year)
    file1 = PATH_IN1 / filename
    file2 = PATH_IN2 / filename
    outfile = PATH_OUT / filename

    ndays1 = get_ndays(file1)
    ndays2 = get_ndays(file2)

    if ndays1 != ndays2:
        raise ValueError(
            f"Input record counts differ for {name} in {year}:\n"
            f"{file1}: {ndays1} days\n"
            f"{file2}: {ndays2} days"
        )

    shape = (ndays1, NY, NX)
    src1 = np.memmap(file1, dtype=DTYPE, mode="r", shape=shape)
    src2 = np.memmap(file2, dtype=DTYPE, mode="r", shape=shape)
    out = np.memmap(outfile, dtype=DTYPE, mode="w+", shape=shape)

    try:
        for y0 in range(0, NY, CHUNK_Y):
            y1 = min(y0 + CHUNK_Y, NY)
            result = operation(
                src1[:, y0:y1, :],
                src2[:, y0:y1, :],
                dtype=np.float32,
            )
            out[:, y0:y1, :] = result

        out.flush()
    finally:
        src1._mmap.close()
        src2._mmap.close()
        out._mmap.close()

    print(f"{name} {year}: {ndays1} days")


# ============================================================
# Run
# ============================================================

PATH_OUT.mkdir(parents=True, exist_ok=True)

symbol = "+" if OPERATION == "add" else "-"
print(f"Operation: field1 {symbol} field2")
print(f"Field 1: {PATH_IN1}")
print(f"Field 2: {PATH_IN2}")
print(f"Output:  {PATH_OUT}")
print(f"Variables: {', '.join(VARIABLES)}")

for variable_name, filename_template in VARIABLES.items():
    for current_year in YEARS:
        process_file(variable_name, filename_template, current_year)

print("All done.")
