import json
import os
import itertools
from typing import Any


REQUIRED_KEYS = {"layer_dims", "activation", "optimizer", "seed", "epochs", "lr", "batch_size"}

AUTOENCODER_REGISTRY: dict[str, tuple[str, str]] = {
    "simple": ("autoencoders.SimpleAutoencoder", "SimpleAutoencoder"),
    "denoising": ("autoencoders.DenoisingAutoencoder", "DenoisingAutoencoder"),
}

def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a JSON config file for the autoencoder pipeline
    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the file is not valid JSON or required keys are missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path!r}")

    with open(config_path, "r", encoding="utf-8") as fh:
        try:
            cfg: dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in config file {config_path!r}: {exc}"
            ) from exc

    # normalise top-level keys to lower-case.
    cfg = {k.lower(): v for k, v in cfg.items()}

    # normalise grid keys if present.
    if "grid" in cfg and isinstance(cfg["grid"], dict):
        cfg["grid"] = {k.lower(): v for k, v in cfg["grid"].items()}

    grid_keys = set(cfg.get("grid", {}).keys())
    flat_keys = cfg.keys() - {"grid"}
    all_keys  = flat_keys | grid_keys

    missing = REQUIRED_KEYS - all_keys
    if missing:
        raise ValueError(
            f"Config file {config_path!r} is missing required key(s): "
            + ", ".join(sorted(missing))
        )

    return cfg


def expand_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a config that may contain a ``"grid"`` section into a flat list
    of fully-resolved parameter dictionaries, one per combination.

    The ``"grid"`` key (if present) must be a mapping whose values are lists
    of candidate values.  Every other top-level key is treated as a fixed
    scalar copied into every combination unchanged.

    If there is no ``"grid"`` key — or the grid is empty — the function
    returns a single-element list containing the config itself (with the
    ``"grid"`` key removed), so callers can always iterate unconditionally.

    """
    grid: dict[str, list] = cfg.get("grid", {})

    # Base config: everything except the "grid" key itself.
    base = {k: v for k, v in cfg.items() if k != "grid"}

    if not grid:
        return [base]

    # Validate that every grid value is a list.
    bad = [k for k, v in grid.items() if not isinstance(v, list)]
    if bad:
        raise ValueError(
            f"Each grid value must be a list. Non-list grid key(s): {bad}"
        )

    grid_keys   = list(grid.keys())
    grid_values = [grid[k] for k in grid_keys]

    combinations: list[dict[str, Any]] = []
    for combo_values in itertools.product(*grid_values):
        entry = dict(base)                          # copy fixed params
        for key, val in zip(grid_keys, combo_values):
            entry[key] = val                        # override / add grid param
        combinations.append(entry)

    return combinations


def resolve_autoencoder(autoencoder_type: str):
    """Return the autoencoder *class* that corresponds to *autoencoder_type*. Case insensitive
    """
    key = autoencoder_type.lower()
    if key not in AUTOENCODER_REGISTRY:
        known = ", ".join(f'"{k}"' for k in sorted(AUTOENCODER_REGISTRY))
        raise ValueError(
            f"Unknown autoencoder type {autoencoder_type!r}. "
            f"Known types: {known}"
        )

    module_path, class_name = AUTOENCODER_REGISTRY[key]

    try:
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ImportError(
            f"Could not import {class_name!r} from {module_path!r}: {exc}"
        ) from exc

    return cls