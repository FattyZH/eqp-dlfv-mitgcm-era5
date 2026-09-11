"""Project paths independent of the caller's working directory."""

import os
from pathlib import Path


def project_root() -> Path:
    """Use WORK_DIR when set; otherwise locate this editable source checkout."""
    configured = os.environ.get("WORK_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    for root in Path(__file__).resolve().parents:
        if (root / "code").is_dir() and (root / "input").is_dir():
            return root
    raise RuntimeError("Set WORK_DIR to the experiment root, or install this project editable.")
