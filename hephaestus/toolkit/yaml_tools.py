"""YAML tools for Hephaestus."""

import os
from pathlib import Path
from typing import Mapping, Any
from ruamel.yaml import YAML

yaml_safe = YAML(typ="safe")
yaml_safe.default_flow_style = False
yaml_safe.default_style = "|"  # type: ignore
yaml_safe.allow_unicode = True

yaml = YAML()
yaml.default_flow_style = False
yaml.default_style = "|"  # type: ignore
yaml.allow_unicode = True


def save_yaml(data: Mapping[str, Any], location: Path) -> None:
    """Save YAML to a file, making sure the directory exists."""
    if not location.exists():
        os.makedirs(location.parent, exist_ok=True)
    yaml.dump(data, location)
