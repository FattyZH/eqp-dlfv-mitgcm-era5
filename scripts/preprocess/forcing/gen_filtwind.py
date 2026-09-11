from mitkit.paths import project_root
from pathlib import Path
import calendar
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

# ============================================================
# Settings
# ============================================================

BASE_DIR = project_root()
EXF_DIR = BASE_DIR / 'data/exf'
PATH_IN = EXF_DIR / "era5_dy"

YEARS = range(1990, 2027)

NX = 761
NY = 241

DTYPE = ">f4"                       # MITgcm big-endian float32
CHUNK_Y = 241

# 18-month band
PATH_OUT = EXF_DIR / "wind_bp24-96"
FILTER_TYPE = "bandpass"           # "bandpass" or "bandstop"
TMIN = 24 * 365.25 / 12                 # days
TMAX = 96 * 365.25 / 12                 # days
ORDER = 6

VARIABLES = {
    "uwind": "uwind_{year}",
    "vwind": "vwind_{year}",
}

# ============================================================
# Butterworth filter
# ============================================================

if FILTER_TYPE not in {"bandpass", "bandstop"}:
    raise ValueError(
        f"Invalid FILTER_TYPE: {FILTER_TYPE!r}; "
        "choose 'bandpass' or 'bandstop'"
    )

fs = 1.0                            # daily data: sample/day

sos = butter(
    ORDER,
    [1 / TMAX, 1 / TMIN],
    btype=FILTER_TYPE,
    fs=fs,
    output="sos",
)

# ============================================================
# Basic information
# ============================================================

YEARS = list(YEARS)
LAST_YEAR = YEARS[-1]

bytes_per_day = NY * NX * np.dtype(DTYPE).itemsize

# 先设置完整年份
ndays = {
    year: 366 if calendar.isleap(year) else 365
    for year in YEARS
}

# 最后一年根据实际文件大小确定天数
last_file = PATH_IN / VARIABLES["uwind"].format(year=LAST_YEAR)
last_size = last_file.stat().st_size

if last_size % bytes_per_day != 0:
    raise ValueError(
        f"{last_file}\n"
        f"File size is not an integer number of daily records.\n"
        f"Actual: {last_size} bytes\n"
        f"Bytes per day: {bytes_per_day}"
    )

ndays[LAST_YEAR] = last_size // bytes_per_day

max_days = 366 if calendar.isleap(LAST_YEAR) else 365
if not 1 <= ndays[LAST_YEAR] <= max_days:
    raise ValueError(
        f"{last_file}: invalid number of daily records "
        f"for {LAST_YEAR}: {ndays[LAST_YEAR]}"
    )

nt = sum(ndays.values())

PATH_OUT.mkdir(parents=True, exist_ok=True)

print(f"Filter type: {FILTER_TYPE}")
print(f"Output directory: {PATH_OUT}")
print(f"{LAST_YEAR}: {ndays[LAST_YEAR]} days")
print(f"Total records: {nt}")

# ============================================================
# File-size check
# ============================================================

def check_file(file, year):
    actual = file.stat().st_size

    if actual % bytes_per_day != 0:
        raise ValueError(
            f"{file}\n"
            f"File size is not an integer number of daily records.\n"
            f"Actual: {actual} bytes\n"
            f"Bytes per day: {bytes_per_day}"
        )

    actual_days = actual // bytes_per_day

    if actual_days != ndays[year]:
        raise ValueError(
            f"{file}\n"
            f"Expected: {ndays[year]} days\n"
            f"Actual:   {actual_days} days"
        )

# ============================================================
# Process one forcing variable
# ============================================================

def process_variable(name, template):

    print(f"\n{'=' * 60}")
    print(f"Processing {name}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # 1. Open input files
    # --------------------------------------------------------

    src = {}

    for year in YEARS:
        file = PATH_IN / template.format(year=year)

        check_file(file, year)

        src[year] = np.memmap(
            file,
            dtype=DTYPE,
            mode="r",
            shape=(ndays[year], NY, NX),
        )

    # --------------------------------------------------------
    # 2. Create output files
    # --------------------------------------------------------

    out = {}

    for year in YEARS:

        shape = (ndays[year], NY, NX)
        filename = template.format(year=year)

        out[year] = np.memmap(
            PATH_OUT / filename,
            dtype=DTYPE,
            mode="w+",
            shape=shape,
        )

    # --------------------------------------------------------
    # 3. Process by y chunks
    # --------------------------------------------------------

    for y0 in range(0, NY, CHUNK_Y):

        y1 = min(y0 + CHUNK_Y, NY)
        ny_chunk = y1 - y0

        print(f"{name}: y = {y0:4d}:{y1:4d} / {NY}")

        data = np.empty(
            (nt, ny_chunk, NX),
            dtype=np.float32,
        )

        i0 = 0

        for year in YEARS:
            i1 = i0 + ndays[year]

            data[i0:i1] = src[year][:, y0:y1, :]

            i0 = i1

        # ----------------------------------------------------
        # Flatten horizontal dimensions
        # ----------------------------------------------------

        shape = data.shape
        data = data.reshape(nt, -1)

        # ----------------------------------------------------
        # Apply the selected filter
        # ----------------------------------------------------

        filtered = sosfiltfilt(
            sos,
            data,
            axis=0,
        )

        filtered = filtered.astype(np.float32, copy=False)

        # ----------------------------------------------------
        # Restore horizontal dimensions
        # ----------------------------------------------------

        filtered = filtered.reshape(shape)

        # ----------------------------------------------------
        # 4. Split by year and write
        # ----------------------------------------------------

        i0 = 0

        for year in YEARS:
            i1 = i0 + ndays[year]

            out[year][:, y0:y1, :] = filtered[i0:i1]

            i0 = i1

        del data, filtered

    # --------------------------------------------------------
    # 5. Close memmaps
    # --------------------------------------------------------

    for year in YEARS:
        src[year]._mmap.close()

        out[year].flush()
        out[year]._mmap.close()

    print(f"{name} done.")

# ============================================================
# Run
# ============================================================

for name, template in VARIABLES.items():
    process_variable(name, template)

print("\nAll done.")
