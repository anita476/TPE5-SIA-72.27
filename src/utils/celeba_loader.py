"""CelebA (CelebFaces) loader.

Reads aligned face JPGs from data/celeba/img_align_celeba/. If the folder does
not yet hold enough images, it downloads a *small subset* on demand from the
public Hugging Face mirror of CelebA (nielsr/CelebA-faces) via the
datasets-server `rows` API

Faces are center-cropped, converted to grayscale and resized to IMG_SIZE x
IMG_SIZE, then flattened -- same shape the rest of the pipeline expects.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
from PIL import Image

# 40x40 grayscale -> 1600-dim vectors (same scale as the B&W emoji experiment).
ROWS, COLS = 40, 40

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "celeba")
_IMG_SUBDIR = "img_align_celeba"
_ATTR_FILE = "list_attr_celeba.txt"

# Public HF mirror of CelebA aligned faces (178x218), served per-image over
_HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_HF_DATASET = "nielsr/CelebA-faces"
_HF_PAGE = 100  # rows per request


def _count_jpgs(img_dir: str) -> list[str]:
    if not os.path.isdir(img_dir):
        return []
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))


def _fetch_rows(offset: int, length: int) -> list[dict]:
    """Fetch a page of CelebA rows (image src URLs) from the HF datasets-server."""
    params = (
        f"?dataset={_HF_DATASET}&config=default&split=train"
        f"&offset={offset}&length={length}"
    )
    req = urllib.request.Request(_HF_ROWS_URL + params,
                                 headers={"User-Agent": "celeba-loader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["rows"]


def _ensure_images(cache_dir: str, n_needed: int) -> str:
    """Return the dir of aligned JPGs, downloading a subset from HF if short."""
    img_dir = os.path.join(cache_dir, _IMG_SUBDIR)
    existing = _count_jpgs(img_dir)
    if len(existing) >= n_needed:
        return img_dir

    os.makedirs(img_dir, exist_ok=True)
    have = len(existing)
    print(f"  Downloading {n_needed - have} CelebA images from {_HF_DATASET} "
          f"(have {have}) ...")
    offset = have
    while have < n_needed:
        length = min(_HF_PAGE, n_needed - have)
        try:
            rows = _fetch_rows(offset, length)
        except Exception as e:
            raise RuntimeError(
                f"Could not download CelebA subset from HF ({e}). "
                f"Place aligned JPGs manually in {img_dir}/."
            )
        if not rows:
            break  # ran out of dataset
        for row in rows:
            src = row["row"]["image"]["src"]
            dest = os.path.join(img_dir, f"{have + 1:06d}.jpg")
            try:
                urllib.request.urlretrieve(src, dest)
                have += 1
            except Exception as e:
                print(f"    skip image (download failed: {e})")
            offset += 1
        print(f"    {have}/{n_needed}")

    if have == 0:
        raise RuntimeError(f"No CelebA images available in {img_dir}/")
    return img_dir


def _load_male_attr(cache_dir: str) -> dict[str, int] | None:
    """Parse a local list_attr_celeba.txt -> {filename: 0|1} for 'Male'.

    Returns None if the file is missing or unparseable (then a single 'face'
    class is used). The HF subset has no attributes, so this only kicks in if
    you supply the official attribute file yourself.
    """
    attr_path = os.path.join(cache_dir, _ATTR_FILE)
    if not os.path.isfile(attr_path):
        return None
    try:
        with open(attr_path, "r", encoding="utf-8") as f:
            _count = f.readline()
            names = f.readline().split()
            male_idx = names.index("Male")
            mapping = {}
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                mapping[parts[0]] = 1 if parts[1 + male_idx] == "1" else 0
        return mapping
    except Exception as e:
        print(f"  Could not parse {_ATTR_FILE} ({e}); using a single class.")
        return None


def _load_image(path: str, img_size: int) -> np.ndarray:
    """Center-crop to square, grayscale, resize to img_size, return flat [0,1]."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        im = im.crop((left, top, left + side, top + side))
        im = im.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_celeba(
    n_samples: int = 2000,
    seed: int = 0,
    cache_dir: str = _CACHE_DIR,
    img_size: int = ROWS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a subsample of CelebA as grayscale vectors.

    Returns (X, class_ids, label_names) where
        X:           (n_samples, img_size*img_size) float32 in [0, 1]
        class_ids:   (n_samples,) int  -- 0/1 by 'Male' attr, or all 0
        label_names: ["face"] or ["Female", "Male"]
    """
    img_dir = _ensure_images(cache_dir, n_samples)
    files = _count_jpgs(img_dir)
    if not files:
        raise RuntimeError(f"No .jpg files found in {img_dir}/")

    rng = np.random.default_rng(seed)
    n = min(n_samples, len(files))
    chosen = rng.choice(len(files), size=n, replace=False)
    chosen_files = [files[i] for i in chosen]

    X = np.stack([
        _load_image(os.path.join(img_dir, f), img_size) for f in chosen_files
    ])

    male = _load_male_attr(cache_dir)
    if male is not None:
        class_ids = np.array([male.get(f, 0) for f in chosen_files], dtype=int)
        label_names = ["Female", "Male"]
    else:
        class_ids = np.zeros(n, dtype=int)
        label_names = ["face"]

    return X, class_ids, label_names
