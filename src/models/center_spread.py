"""Recentre and rescale predictions onto a target mean and standard deviation per isoform.

Given a target mean and spread for an isoform's labels, this maps a set of predictions onto them:

    p' = mean + (std / sd(p)) * (p - mean(p))

so the transformed predictions carry exactly that mean and standard deviation. This is moment
matching, not a fit: no labels are used, only the two numbers supplied per isoform.

It is a different objective from the least-squares map in `recalibrate.py`, and a worse one for
squared error. The MSE-optimal affine map shrinks the spread to `r * std` (r being the correlation),
while this keeps it at `std`, which costs a factor of `2 / (1 + r)` in MSE. What it buys instead is an
exact mean and spread, which is worth having when those are known for the set being predicted and
differ from the training distribution the model regressed toward. Where the numbers come from (a
held-out set, prior knowledge, a published summary) is the caller's business, and their accuracy is
the whole of its value.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: The CSV must have these columns; `isoform` takes either `3A4` or the full target column name.
COLUMNS = ("isoform", "mean", "std")

_SUFFIX = "_pIC50_direct_inhibition"


def _target_name(isoform: str) -> str:
    """The full target column an `isoform` cell names, expanding the `3A4` and `CYP3A4` shorthands."""
    name = isoform.strip()
    if name.endswith(_SUFFIX):
        return name
    return f"{name if name.upper().startswith('CYP') else 'CYP' + name}{_SUFFIX}".replace("cyp", "CYP", 1)


def load_targets(path: Path) -> dict[str, tuple[float, float]]:
    """Read `isoform,mean,std` rows into {target column: (mean, std)}.

    Raises ValueError naming the problem for a missing column, an unreadable number, a non-positive
    standard deviation, or the same isoform twice.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise ValueError(f"Cannot read the centre/spread CSV {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} has no rows; expected the columns {', '.join(COLUMNS)}")
    missing = [c for c in COLUMNS if c not in (rows[0].keys() or {})]
    if missing:
        raise ValueError(f"{path} has no column(s) {', '.join(missing)}; expected {', '.join(COLUMNS)}")
    targets: dict[str, tuple[float, float]] = {}
    for number, row in enumerate(rows, start=2):  # row 1 is the header
        target = _target_name(row["isoform"] or "")
        try:
            mean, std = float(row["mean"]), float(row["std"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} line {number}: mean and std must be numbers, got {row['mean']!r} and {row['std']!r}") from exc
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"{path} line {number}: mean and std must be finite, got {mean} and {std}")
        if std <= 0:
            raise ValueError(f"{path} line {number}: std must be above zero, got {std}")
        if target in targets:
            raise ValueError(f"{path} line {number}: {target} appears twice")
        targets[target] = (mean, std)
    return targets


def center_spread(values: np.ndarray, mean: float, std: float, source: np.ndarray | None = None) -> np.ndarray:
    """Map `values` onto `mean` and `std`, measuring the spread to divide out on `source` (default: `values`).

    Passing `source` lets several related sets share one scaling, e.g. every cross-validation fold
    scaled by the pooled out-of-fold predictions rather than each fold's own handful of rows.
    Predictions that are all one number have no spread to rescale, so they are only shifted.
    """
    values = np.asarray(values, dtype=float)
    source = values if source is None else np.asarray(source, dtype=float)
    spread = float(source.std())
    if spread == 0.0:
        return values - float(source.mean()) + mean
    return mean + (std / spread) * (values - float(source.mean()))
