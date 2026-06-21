"""training_run.py
Single-combination training worker.

the caller (grid_runner) only needs the
returned :class:`RunResult`.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from utils.config_loader import resolve_autoencoder


@dataclass
class RunResult:
    """All outputs produced by a single training run."""
    run_id:           int
    params:           dict[str, Any]          # flat hyperparameters used
    epoch_losses:     list[float]             # mean batch loss per epoch
    passed:           int
    failed:           int
    avg_pixel_errors: float
    max_pixel_errors: float
    labels:           list[str]      = field(default_factory=list)
    per_char_errors:  np.ndarray     = field(default_factory=lambda: np.array([]))
    reconstruction_path: str         = ""
    plot_path:           str         = ""
    latent:             Any = None
    X:                  Any = None
    label:              Any = None



def _binarise(output: np.ndarray, threshold: float) -> np.ndarray:
    return (output >= threshold).astype(np.float32)


def _pixel_errors(pred_bin: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.sum(pred_bin != y_true, axis=1)


def _format_bitmap_side_by_side(
    original: np.ndarray,
    reconstructed: np.ndarray,
    label: str,
    errors: int,
    rows: int = 7,
    cols: int = 5,
) -> str:
    orig_grid = original.reshape(rows, cols)
    rec_grid  = reconstructed.reshape(rows, cols)
    lines = [f"[ {label} ]  —  {errors} pixel error(s)"]
    lines.append(f"  {'Original':<13}  {'Reconstructed'}")
    lines.append(f"  {'-'*11}  {'-'*13}")
    for r in range(rows):
        orig_row = " ".join("█" if p else "·" for p in orig_grid[r])
        rec_row  = " ".join("█" if p else "·" for p in rec_grid[r])
        diff     = "  ✗" if not np.array_equal(orig_grid[r], rec_grid[r]) else ""
        lines.append(f"  {orig_row}   {rec_row}{diff}")
    lines.append("")
    return "\n".join(lines)



def execute_run(
    run_id:    int,
    params:    dict[str, Any],
    X:         np.ndarray,
    labels:    list[str],
    out_dir:   str,
) -> RunResult:
    """Train one autoencoder with *params* and write per-run output files.

    Parameters
    ----------
    run_id:
        1-based index used for naming output files.
    params:
        Flat hyperparameter dict for this combination.
    X:
        Input data array ``(n_samples, n_features)``.
    labels:
        Character labels matching rows of *X*.
    out_dir:
        Directory where ``reconstruction_<run_id>.txt`` will be written.
        Must already exist (created by the caller).

    Returns
    -------
    RunResult
        All metrics and paths produced by this run.
    """
    autoencoder_type = params.get("autoencoder_type", "simple")
    layer_dims       = params["layer_dims"]
    activation       = params["activation"]
    optimizer        = params["optimizer"]
    seed             = params["seed"]
    epochs           = params["epochs"]
    lr               = params["lr"]
    batch_size       = params["batch_size"]
    log_every        = params.get("log_every", 0)
    threshold        = params.get("threshold", 0.5)
    max_errors       = params.get("max_errors", 1)
    weight_init      = params.get("weight_init", params.get("init", "he"))
    patience         = params.get("patience")
    min_delta        = params.get("min_delta", 1e-6)
    write_report     = params.get("write_report", True)

    prefix = f"[run {run_id}]"
    print(f"{prefix} autoencoder={autoencoder_type}  activation={activation}  "
          f"init={weight_init}  lr={lr}  batch={batch_size}  seed={seed}  epochs={epochs}")

    AEClass = resolve_autoencoder(autoencoder_type)
    ae = AEClass(layer_dims, activation, seed, weight_init=weight_init)
    # Optional output loss ("mse"/"bce") for autoencoders that support it.
    if "loss" in params and hasattr(ae, "loss_type"):
        ae.loss_type = params["loss"]

    epoch_losses = ae.train_and_collect(
        X, epochs, lr, batch_size, log_every, optimizer,
        patience=patience, min_delta=min_delta,
    )

    latent_all        = ae.encode(X)
    reconstructed_raw = ae.decode(latent_all)
    reconstructed_bin = _binarise(reconstructed_raw, threshold)
    errors            = _pixel_errors(reconstructed_bin, X)

    passed  = int(np.sum(errors <= max_errors))
    failed  = len(errors) - passed
    avg_err = float(errors.mean())
    max_err = float(errors.max())

    print(f"{prefix} done — passed {passed}/{len(X)}, "
          f"avg_err={avg_err:.2f}, max_err={max_err}")

    # write reconstruction.txt (skipped in multi-seed runs to avoid clutter)
    reconstruction_path = ""
    if write_report:
        reconstruction_path = os.path.join(out_dir, f"reconstruction_{run_id}.txt")
        with open(reconstruction_path, "w", encoding="utf-8") as f:
            f.write(f"Autoencoder Reconstruction Report  [run {run_id}]\n")
            f.write(f"Autoencoder  : {autoencoder_type} ({AEClass.__name__})\n")
            f.write(f"Architecture : {layer_dims}\n")
            f.write(f"Activation   : {activation}  |  Optimizer : {optimizer}  |  Init : {weight_init}\n")
            f.write(f"LR: {lr}  |  Batch: {batch_size}  |  Seed: {seed}\n")
            f.write(f"Threshold : {threshold}  |  Epochs: {epochs}\n")
            f.write(f"Passed: {passed}/{len(X)}  |  "
                    f"Avg errors: {avg_err:.2f}  |  Max errors: {max_err}\n")
            f.write("=" * 50 + "\n\n")
            for i, (label, err) in enumerate(zip(labels, errors)):
                block = _format_bitmap_side_by_side(X[i], reconstructed_bin[i], label, int(err))
                f.write(block + "\n")

    return RunResult(
        run_id           = run_id,
        params           = params,
        epoch_losses     = epoch_losses,
        passed           = passed,
        failed           = failed,
        avg_pixel_errors = avg_err,
        max_pixel_errors = max_err,
        labels           = list(labels),
        per_char_errors  = errors,
        reconstruction_path = reconstruction_path,
        plot_path        = "",          # plot is produced by grid_runner after collect
        latent = latent_all,
        X = X,
        label = labels
    )