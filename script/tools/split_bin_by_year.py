from pathlib import Path

import numpy as np


# ====== parameters to edit ======
input_files = [
    "../../data/exf/era5_dc/uwind.1990",
    "../../data/exf/era5_dc/vwind.1990",
]
output_dir = "../../data/exf/era5_dy"
start_date = "1990-01-01"
ny, nx = 241, 761
overwrite = False
# ================================


out_dir = Path(output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for input_file in input_files:
    input_path = Path(input_file)
    values_per_day = ny * nx
    nt = input_path.stat().st_size // 4 // values_per_day

    a = np.memmap(input_path, dtype=">f4", mode="r")
    b = a.reshape((nt, ny, nx))
    t = np.arange(nt) + np.datetime64(start_date)
    years = t.astype("datetime64[Y]").astype(int) + 1970

    print(input_path, b.shape, t[0], t[-1])

    for year in np.unique(years):
        out_path = out_dir / f"{input_path.stem}_{year}"
        if out_path.exists() and not overwrite:
            print(f"skip existing: {out_path}")
            continue

        one_year = b[years == year]
        one_year.astype(">f4").tofile(out_path)
        print(f"write: {out_path}, shape={one_year.shape}")
