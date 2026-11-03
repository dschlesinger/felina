"""Affine (linear) recalibration of regression predictions.

Fits ``y = slope * prediction + intercept`` by least squares on held-out predictions, the regression
analogue of Platt scaling (Platt 1999; Guo et al. 2017 "On calibration of modern neural networks")
and the "calibration slope / calibration-in-the-large" correction of clinical prediction models
(Steyerberg et al. 2010, Epidemiology 21:128). Kuleshov et al. 2018 ("Accurate uncertainties for deep
learning using calibrated regression") recalibrate with the same fit-on-held-out protocol. A slope
below one shrinks over-dispersed predictions, above one stretches the compressed ones that
tree/ensemble averages produce, and the intercept removes a global bias.

The fit uses only out-of-fold predictions, never rows a model was trained on, and is applied once.
"""

from __future__ import annotations

import numpy as np


def fit_affine(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Least-squares (slope, intercept) mapping out-of-fold `predicted` onto `actual`.

    Falls back to the identity when the predictions are constant or there are fewer than two rows,
    where the slope is not identifiable.
    """
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(actual, dtype=float)
    if len(p) < 2 or np.ptp(p) == 0.0:
        return 1.0, 0.0
    slope, intercept = np.polyfit(p, y, 1)
    return float(slope), float(intercept)


def apply_affine(predicted: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply a fitted affine map to predictions."""
    return slope * np.asarray(predicted, dtype=float) + intercept


def recalibrate(out: dict, y: np.ndarray, folds: np.ndarray) -> tuple[dict, tuple[float, float]]:
    """Affine-recalibrate one prediction set ({"val", "blind", "holdout"}) on its out-of-fold predictions.

    The map is fitted on every fold's held-out predictions and applied to the validation and blind
    rows. Each fold's own held-out predictions are recalibrated with a map fitted on the *other* folds
    only, so cross-validation metrics on the result are not scored on rows the map was fitted on.
    Returns the recalibrated set and the (slope, intercept) applied to val/blind.
    """
    pooled = np.empty(len(y))
    for k, pred in out["holdout"].items():
        pooled[folds == k] = pred
    slope, intercept = fit_affine(pooled, y)
    holdout = {}
    for k, pred in out["holdout"].items():
        rest = folds != k
        s, b = fit_affine(pooled[rest], y[rest])
        holdout[k] = apply_affine(pred, s, b)
    recalibrated = {
        "val": apply_affine(out["val"], slope, intercept),
        "blind": None if out["blind"] is None else apply_affine(out["blind"], slope, intercept),
        "holdout": holdout,
    }
    return recalibrated, (slope, intercept)
