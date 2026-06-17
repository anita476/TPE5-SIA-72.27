"""Shared bootstrap for the entry-point scripts.

Importing this module makes the ``src`` package importable regardless of the
current working directory, and exposes ``resolve`` to turn config-relative
paths into absolute ones anchored at the project root (so outputs always land
in ``<root>/output`` no matter where the script is launched from).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve(path: str) -> str:
    """Return *path* as an absolute path anchored at the project root."""
    p = Path(path)
    return str(p if p.is_absolute() else (ROOT / p))
