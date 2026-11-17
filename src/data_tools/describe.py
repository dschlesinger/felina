"""Shared pieces for `data explore` and `data compare`: table resolution, named features, effect sizes.

Both commands answer "what is in this data, and how does it differ from that data", and both answer it in
terms a chemist can act on. That rules out raw fingerprint bits: a Morgan bit that separates two sets names
nothing. Everything here is therefore keyed on *named* features -- RDKit's physicochemical descriptors plus
the project's cited CYP motif blocks -- so a reported difference reads as "the right-hand set is less basic"
rather than "bit 1174 fires more often".

Fingerprints still do the work where names do not matter: clustering, nearest-neighbour distance, and the
two-sample probe. Those live in the two command modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Columns that are identifiers or bookkeeping, never labels, when guessing which targets a table carries.
_NOT_A_TARGET = (
    "smiles",
    "inchikey",
    "inchikey_block",
    "inchikey_full",
    "molecule_name",
    "name",
    "id",
    "split",
    "split_blindlike",
    "fold",
    "source",
    "dataset",
)

_MIN_GROUP = 3  # below this a mean and a standard deviation say nothing


@dataclass(frozen=True)
class FeatureShift:
    """One named feature's difference between a group and its complement.

    `d` is Cohen's d (positive when the group sits higher), `p` the Mann-Whitney U p-value adjusted for the
    number of features tested. The test is rank-based because most of these features are bounded counts whose
    distributions are nowhere near normal; d is kept alongside it because a p-value on 20,000 molecules is
    significant for differences far too small to act on.
    """

    name: str
    d: float
    p: float
    group_mean: float
    rest_mean: float

    def describe(self) -> str:
        """One line: `NumAromaticRings 2.81 vs 1.42 (d=+1.31, p=2e-18)`."""
        return (
            f"{self.name} {self.group_mean:.3g} vs {self.rest_mean:.3g} "
            f"(d={self.d:+.2f}, p={self.p:.1e})"
        )


def resolve_table(
    reference: str,
    data_dir: Path,
    *,
    smiles_column: str = "SMILES",
) -> tuple[pl.DataFrame, str]:
    """Load a table named either by registered dataset (or alias) or by CSV path.

    Returns the frame and a short display name. Rows whose SMILES is null or blank are dropped here, once,
    so neither command has to carry that case through its own logic.
    """
    from data_tools import dataset_alias
    from data_tools.datasets import registry

    datasets = registry()
    path: Path
    try:
        name = dataset_alias.resolve(reference, set(datasets))
        path = data_dir / datasets[name].primary
        if not path.is_file():
            raise ValueError(f"dataset {name!r} is not fetched yet; run `data fetch {name}`")
    except ValueError as unknown:
        candidate = Path(reference)
        if not candidate.is_file():
            # Neither a dataset nor a file: report both failures, since the user meant one of them.
            raise ValueError(f"{unknown} -- and {reference!r} is not a readable file either") from unknown
        path, name = candidate, candidate.stem

    df = pl.read_csv(path, infer_schema_length=None)
    if smiles_column not in df.columns:
        raise ValueError(f"{name}: no {smiles_column!r} column; it has {', '.join(df.columns)}")
    kept = df.filter(
        pl.col(smiles_column).is_not_null() & (pl.col(smiles_column).str.strip_chars() != "")
    )
    return kept, name


def target_columns(df: pl.DataFrame, requested: list[str] | None = None) -> list[str]:
    """The label columns to summarize: the ones asked for, else every numeric non-identifier column.

    Guessing rather than requiring `--target` matters because these commands are pointed at external and
    merged tables as often as at the challenge's own, and those carry label columns under their own names.
    """
    if requested:
        missing = [t for t in requested if t not in df.columns]
        if missing:
            raise ValueError(f"no column(s) {', '.join(missing)}; the table has {', '.join(df.columns)}")
        return list(requested)
    found = []
    for column, dtype in zip(df.columns, df.dtypes):
        if column.casefold() in _NOT_A_TARGET or not dtype.is_numeric():
            continue
        if df[column].drop_nulls().len() >= _MIN_GROUP:
            found.append(column)
    return found


def named_features(smiles: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Return an (n_molecules, n_features) matrix of interpretable features and their names.

    RDKit's 2D descriptors carry the physicochemical axes (size, lipophilicity, polarity, charge-related
    counts); the CYP blocks carry the mechanism-relevant motifs. Unparseable molecules come back as all-NaN
    rows, which every consumer here drops rather than imputes.
    """
    from representations.cyp_pharmacophore import (
        HemeCoordinationMotifs,
        IsoformPreferenceDescriptors,
        MechanismBasedAlerts,
    )
    from representations.rdkit_descriptors import RDKitDescriptors

    blocks = [
        RDKitDescriptors(),
        HemeCoordinationMotifs(),
        MechanismBasedAlerts(),
        IsoformPreferenceDescriptors(),
    ]
    matrices, names = [], []
    for block in blocks:
        matrices.append(block.transform(smiles))
        names.extend(block.feature_names)
    return np.hstack(matrices), names


def fingerprints(smiles: pl.Series, *, counts: bool = True) -> np.ndarray:
    """Morgan fingerprints for clustering, distance and the two-sample probe; all-NaN for bad SMILES."""
    from representations.morgan import CountMorganFingerprint, MorganFingerprint

    block = CountMorganFingerprint() if counts else MorganFingerprint()
    return block.transform(smiles)


def usable_columns(X: np.ndarray) -> np.ndarray:
    """Boolean mask of feature columns that are finite everywhere and not constant.

    A descriptor that is NaN or Inf for even one molecule (RDKit's Ipc overflows on large rings) would
    otherwise poison a standardization or a probe fit for the whole table.
    """
    if X.size == 0:
        return np.zeros(X.shape[1], dtype=bool)
    finite = np.isfinite(X).all(axis=0)
    varying = np.zeros(X.shape[1], dtype=bool)
    if X.shape[0] > 1:
        with np.errstate(invalid="ignore"):
            varying = np.nanstd(X, axis=0) > 0
    return finite & varying


def feature_shifts(
    X: np.ndarray,
    names: list[str],
    mask: np.ndarray,
    *,
    top: int = 5,
    min_abs_d: float = 0.0,
) -> list[FeatureShift]:
    """The `top` features separating `mask` rows from the rest, ranked by |Cohen's d|.

    Returns fewer than `top` entries, or none at all, when either side is too small to compare or no feature
    varies. p-values are Bonferroni-adjusted over the features actually tested, not over all of them: a
    constant column was never a hypothesis.
    """
    from scipy import stats

    mask = np.asarray(mask, dtype=bool)
    group, rest = X[mask], X[~mask]
    if len(group) < _MIN_GROUP or len(rest) < _MIN_GROUP:
        return []

    columns = np.flatnonzero(usable_columns(X))
    if columns.size == 0:
        return []

    shifts: list[FeatureShift] = []
    for j in columns:
        a, b = group[:, j], rest[:, j]
        # Pooled standard deviation; a zero means the two sides are each constant at the same value.
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        if not np.isfinite(pooled) or pooled == 0:
            continue
        d = float((a.mean() - b.mean()) / pooled)
        if abs(d) < min_abs_d:
            continue
        try:
            p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:  # every value identical across both sides
            continue
        shifts.append(FeatureShift(names[j], d, min(1.0, p * columns.size), float(a.mean()), float(b.mean())))

    shifts.sort(key=lambda s: -abs(s.d))
    return shifts[:top]


def target_stats(values: np.ndarray) -> tuple[int, float, float]:
    """(n, mean, std) over the finite entries; NaN mean and std when there are fewer than three."""
    finite = values[np.isfinite(values)]
    if len(finite) < _MIN_GROUP:
        return len(finite), float("nan"), float("nan")
    return len(finite), float(finite.mean()), float(finite.std(ddof=1))
