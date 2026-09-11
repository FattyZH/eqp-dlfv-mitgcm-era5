"""Project paths independent of the caller's working directory."""

import os
from pathlib import Path


def project_root() -> Path:
    """Use WORK_DIR when set; otherwise locate this editable source checkout."""
    configured = os.environ.get("WORK_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Set WORK_DIR to the experiment root, or install this project editable.")
    return root
