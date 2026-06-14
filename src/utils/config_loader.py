import json
import os
from typing import Any


REQUIRED_KEYS = {"layer_dims", "activation", "optimizer", "seed", "epochs", "lr", "batch_size"}

AUTOENCODER_REGISTRY: dict[str, tuple[str, str]] = {
    "simple": ("autoencoders.SimpleAutoencoder", "SimpleAutoencoder"),
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a JSON config file for the autoencoder pipeline.

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
            raise ValueError(f"Invalid JSON in config file {config_path!r}: {exc}") from exc

    cfg = {k.lower(): v for k, v in cfg.items()}

    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(
            f"Config file {config_path!r} is missing required key(s): "
            + ", ".join(sorted(missing))
        )

    return cfg


def resolve_autoencoder(autoencoder_type: str):
    """Return the autoencoder *class* that corresponds to *autoencoder_type*. Case insensitive
    Raises
    ------
    ValueError
        If *autoencoder_type* is not found in the registry.
    ImportError
        If the module or class cannot be imported.
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