#!/usr/bin/env python3
"""
根据 MITgcm 的 data 文件自动选择并安装 pickup 文件。

选择规则：
1. 若 data 中设置了非空 pickupSuff，使用 pickupSuff；
2. 否则读取 nIter0，并格式化为十位迭代后缀；
3. nIter0=0 且没有 pickupSuff 时按冷启动处理；
4. 同时处理主 pickup 和所有 package pickup。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import f90nml


def find_parameter(nml: Mapping[str, Any], name: str) -> Any | None:
    """不区分大小写，在所有 namelist 分组中查找唯一参数。"""
    matches: list[tuple[str, Any]] = []

    def walk(obj: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in obj.items():
            key_str = str(key)
            location = f"{prefix}/{key_str}" if prefix else key_str

            if key_str.lower() == name.lower():
                matches.append((location, value))

            if isinstance(value, Mapping):
                walk(value, location)

    walk(nml)

    if not matches:
        return None

    if len(matches) > 1:
        locations = ", ".join(location for location, _ in matches)
        raise ValueError(f"Parameter {name!r} appears more than once: {locations}")

    return matches[0][1]


def determine_suffix(data_file: Path) -> str | None:
    """从 data 读取 pickupSuff 或 nIter0，返回 pickup 文件后缀。"""
    nml = f90nml.read(data_file)

    pickup_suff = find_parameter(nml, "pickupSuff")
    if pickup_suff is not None:
        suffix = str(pickup_suff).strip()
        if suffix:
            if len(suffix) > 10:
                raise ValueError(
                    f"pickupSuff={suffix!r} exceeds MITgcm's 10-character limit"
                )
            return suffix

    rw_suffix_type = find_parameter(nml, "rwSuffixType")
    if rw_suffix_type is not None and int(rw_suffix_type) != 0:
        raise ValueError(
            "rwSuffixType is not 0, but pickupSuff is empty. "
            "Set pickupSuff explicitly or extend link_pickup.py for this naming mode."
        )

    niter0 = find_parameter(nml, "nIter0")
    if niter0 is None:
        return None

    niter0_int = int(niter0)

    if niter0_int == 0:
        return None

    if niter0_int < 0:
        raise ValueError(f"nIter0 must not be negative: {niter0_int}")

    return f"{niter0_int:010d}"


def find_pickup_files(pickup_dir: Path, suffix: str) -> list[Path]:
    """查找指定后缀的主 pickup 和 package pickup。"""
    files = sorted(
        path
        for path in pickup_dir.glob(f"pickup*.{suffix}*")
        if path.is_file()
    )

    main_pickup = [
        path for path in files
        if path.name.startswith(f"pickup.{suffix}")
    ]

    if not main_pickup:
        raise FileNotFoundError(
            f"No main pickup files matching 'pickup.{suffix}*' "
            f"were found in {pickup_dir}"
        )

    return files


def install_pickup_files(
    files: list[Path],
    dest_dir: Path,
    mode: str,
) -> list[Path]:
    """将 pickup 文件软链接或复制到运行目录。"""
    installed: list[Path] = []

    for source in files:
        target = dest_dir / source.name

        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Destination already contains {target.name}; refusing to overwrite it"
            )

        if mode == "link":
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)

        installed.append(target)

    return installed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read nIter0/pickupSuff from an MITgcm data file and install "
            "the matching pickup files in a run directory."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the MITgcm data namelist",
    )
    parser.add_argument(
        "--pickup-dir",
        type=Path,
        required=True,
        help="Directory containing one or more pickup times",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination run directory",
    )
    parser.add_argument(
        "--mode",
        choices=("link", "copy"),
        default="link",
        help="Install method; default: link",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_file = args.data.expanduser().resolve()
    dest_dir = args.dest.expanduser().resolve()

    if not data_file.is_file():
        raise FileNotFoundError(
            f"MITgcm data file not found: {data_file}"
        )

    if not dest_dir.is_dir():
        raise NotADirectoryError(
            f"Destination directory not found: {dest_dir}"
        )

    # 先读取data，判断是否启用pickup
    suffix = determine_suffix(data_file)

    # nIter0=0且没有pickupSuff：冷启动
    if suffix is None:
        print(
            "[INFO] Cold start: nIter0=0 and pickupSuff is not set. "
            "No pickup files will be installed."
        )
        return 0

    # 只有重启动时才检查pickup目录
    pickup_dir = args.pickup_dir.expanduser()

    if not pickup_dir.is_absolute():
        pickup_dir = pickup_dir.resolve()
    else:
        pickup_dir = pickup_dir.resolve()

    if not pickup_dir.is_dir():
        raise NotADirectoryError(
            f"Pickup is enabled in data, but the pickup directory "
            f"does not exist: {pickup_dir}"
        )

    files = find_pickup_files(pickup_dir, suffix)

    installed = install_pickup_files(
        files=files,
        dest_dir=dest_dir,
        mode=args.mode,
    )

    action = "Linked" if args.mode == "link" else "Copied"

    print(f"[INFO] Restart enabled")
    print(f"[INFO] Pickup suffix: {suffix}")
    print(f"[INFO] Pickup source: {pickup_dir}")
    print(f"[INFO] {action} {len(installed)} pickup files:")

    for target in installed:
        if target.is_symlink():
            source = target.resolve()
        else:
            source = pickup_dir / target.name

        print(f"  {target.name} <- {source}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)