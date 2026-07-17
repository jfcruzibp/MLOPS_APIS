from pathlib import Path
from typing import Any

import yaml


def load_params(path: str | Path) -> dict[str, Any]:
    """Load pipeline parameters from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent(path: str | Path) -> Path:
    """Create the parent folder for an output file and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

