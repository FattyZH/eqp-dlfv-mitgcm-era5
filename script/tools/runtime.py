"""
用法：python mitgcm_efficiency.py [VarName]
"""

import re, sys
from pathlib import Path
import f90nml

def detect_grids(output_dir):
    nml = f90nml.read(output_dir/'data')
    nx = len(nml['PARM04']['delX'])
    ny = len(nml['PARM04']['delY'])
    nr = len(nml['PARM04']['delR'])
    return nx*ny*nr

def detect_ncore(output_dir):
    pattern = re.compile(r"^STDOUT\.(\d+)$")
    ranks = []

    for f in output_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            ranks.append(int(m.group(1)))

    return len(set(ranks)) if ranks else None


def analyze(output_dir):
    pattern = re.compile(r"^(.+?)\.(\d{10})\.meta$")

    groups = {}

    for f in output_dir.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue

        vname = m.group(1)
        it = int(m.group(2))

        groups.setdefault(vname, {})[it] = f

    if not groups:
        return None

    # 自动选择迭代跨度最大的变量
    def span(vname):
        its = groups[vname].keys()
        return max(its) - min(its)

    var = max(groups, key=span)
    files = groups[var]

    if len(files) < 2:
        return None

    first, last = min(files), max(files)

    t0 = files[first].stat().st_mtime
    t1 = files[last].stat().st_mtime

    total_sec = t1 - t0
    avg_sec = total_sec / (last - first)

    return first, last, total_sec, avg_sec, var


def main():
    base = Path.home() / "eqp-dlfv-mitgcm-era5/output"

    subdirs = sorted(d for d in base.iterdir() if d.is_dir())
    if not subdirs:
        print(f"未找到子目录：{base}")
        return

    rows = []

    for d in subdirs:
        result = analyze(d)
        ncore = detect_ncore(d)
        ng = detect_grids(d)
        if result is None:
            rows.append((d.name, "--", "--", "--", "--", "--", "--", "--"))
            continue

        first, last, total_sec, avg_sec, v = result

        iter_per_h = 3600 / avg_sec

        if ncore is None:
            ncore_str = "--"
            iter_per_h_core_str = "--"
            iter_per_h_core_grid_str = "--"
        else:
            ncore_str = str(ncore)
            iter_per_h_core_str = f"{iter_per_h / ncore:.2f}"
            iter_per_h_core_grid_str = f"{iter_per_h * ng / ncore/1e6:.2f}"
        rows.append((
            d.name,
            f"{first}→{last}",
            f"{total_sec / 3600:6.2f}",
            f"{avg_sec:.4f}",
            f"{iter_per_h:.0f}",
            ncore_str,
            iter_per_h_core_str,
            iter_per_h_core_grid_str,
        ))

    headers = (
        "Dir",
        "Iter Range",
        "Total(h)",
        "s/iter",
        "iter/h",
        "nCore",
        "iter/h/core",
        "iter*grid/h/core(M)",
    )

    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))

    for r in rows:
        print(fmt.format(*r))


main()