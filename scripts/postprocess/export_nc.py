from mitkit.paths import project_root
import os
import sys
from pathlib import Path

work_dir = project_root()

from mitkit.io import open_mds

exp = [
    "260810_142313_ctrl1",
    "260810_143059_rb_30d",
    "260810_153834_bp",
    "260810_153853_bs",
]

VAR = ["UVEL"]   # 也可以写 "UVEL"

exps = [exp] if isinstance(exp, str) else exp
vars_ = [VAR] if isinstance(VAR, str) else VAR

# 例如 ["UVEL", "VVEL"] -> "UV"
var_tag = "".join(var[0] for var in vars_)

outdir = work_dir / "output_nc"
outdir.mkdir(parents=True, exist_ok=True)

for exp_name in exps:
    inpath = work_dir / "output" / exp_name
    outpath = outdir / f"{exp_name}_{var_tag}.nc"

    print(f"Processing: {exp_name}")

    ds = open_mds(inpath, prefix="diag3d")
    ds_out = ds[vars_]
    ds_out.to_netcdf(
        outpath,
        engine="netcdf4",
    )

    ds.close()
    print(f"Saved: {outpath}")
