from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    config["_project_dir"] = str(config_path.parents[1])
    return config


def resolve_path(value: str | None, *, base_dir: str | Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base_dir) / path).resolve()


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value
