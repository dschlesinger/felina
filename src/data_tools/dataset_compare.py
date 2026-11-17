"""`data compare`: measure how far one dataset sits from another, and say what the gap is made of.

The question behind every use of this is the same: *if I train on A, what happens on B?* Six readings answer
it from different directions, and they are reported together because any one of them alone misleads.

  overlap        shared molecules on the InChIKey connectivity block. A non-zero overlap with a blind set is
                 leakage; a non-zero overlap between train and an external source is duplication.

  probe AUC      cross-validated AUROC of a classifier told to separate A from B on fingerprints. 0.5 means
                 the two are indistinguishable; 1.0 means a model can always tell which set a molecule came
                 from, and anything trained on A is extrapolating on B.

  the null       the same probe run on A split against *itself* at random. This is the reading the headline
                 AUC is worthless without: with a few hundred molecules in thousands of dimensions a probe
                 separates random halves of one dataset well above 0.5, so an AUC of 0.72 against a null of
                 0.70 is nothing. `auc_gap` is the honest number.

  distance       nearest-neighbour Tanimoto from each B molecule to its closest A molecule, and `nn_ratio`,
                 that distance over how far A sits from itself. This is the axis the CYP blind set was
                 mis-read on: scaffold-held-out molecules sat at median Tanimoto 0.435 from train while the
                 real blind set sat at 0.578, so scaffold CV answered a harder question than the one asked.

  drivers        the named features whose distributions move most between the two sets, by effect size. The
                 probe says *whether* they differ; this says along which axis, which is the part you can act
                 on -- a basicity shift between train and test is a reason to add a pKa feature.

  labels         per shared target, each side's mean and spread and a Kolmogorov-Smirnov test. Two sets can
                 be chemically alike and still carry labels on different scales, which is an assay
                 difference, not a chemistry one, and needs recalibration rather than more features.

Coverage closes it: the share of B's scaffolds and B's clusters that A contains at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.describe import FeatureShift, feature_shifts, named_features, target_stats, usable_columns

_MAX_PROBE = 3000  # rows sampled per side for the probe, MMD and distance work
_MIN_PROBE = 5  # the probe stratifies into five folds, so fewer than five a side is no reading at all
_DEFAULT_CUTOFF = 0.4


@dataclass(frozen=True)
class LabelShift:
    """One target's distribution on each side, with a distribution test."""

    name: str
    left_n: int
    left_mean: float
    left_std: float
    right_n: int
    right_mean: float
    right_std: float
    ks: float
    p: float

    def describe(self, left: str, right: str, suffix: str = "") -> str:
        label = self.name.removesuffix(suffix) if suffix else self.name
        return (
            f"{label}: {left} {self.left_mean:.2f}+/-{self.left_std:.2f} (n={self.left_n})  "
            f"{right} {self.right_mean:.2f}+/-{self.right_std:.2f} (n={self.right_n})  "
            f"KS={self.ks:.3f} p={self.p:.1e}"
        )


@dataclass(frozen=True)
class Comparison:
    """Everything `compare` measured, in the order it is reported."""

    left_name: str
    right_name: str
    left_rows: int
    right_rows: int
    left_keys: int
    right_keys: int
    shared_keys: int

    probe_auc: float
    probe_auc_null: float
    descriptor_auc: float
    probe_drivers: tuple[tuple[str, float], ...]  # (feature, standardized logistic coefficient)

    nn_similarity_median: float  # median Tanimoto of each right molecule to its nearest left molecule
    nn_ratio: float
    frac_out: float
    mmd: float
    mmd_p: float

    feature_drivers: tuple[FeatureShift, ...]
    label_shifts: tuple[LabelShift, ...] = ()
    scaffold_coverage: float = float("nan")
    cluster_coverage: float = float("nan")
    label_suffix: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def auc_gap(self) -> float:
        """How much of the probe's separation is real rather than the null's small-sample inflation."""
        return self.probe_auc - self.probe_auc_null

    @property
    def shared_pct_left(self) -> float:
        return 100.0 * self.shared_keys / self.left_keys if self.left_keys else 0.0

    @property
    def shared_pct_right(self) -> float:
        return 100.0 * self.shared_keys / self.right_keys if self.right_keys else 0.0

    def verdict(self) -> str:
        """One line naming what the numbers add up to, read across both axes rather than one.

        Separability and distance are independent, and the combination that catches people out is high
        separability with *low* distance: the two sets are distinguishable populations, yet every right-hand
        molecule has a close left-hand neighbour. That is the analog-expansion regime, and it is why
        scaffold- or cluster-held-out validation reports a harder problem than the one actually being
        scored. Reading the probe alone would call that case "strongly shifted" and send you to exactly the
        split that overstates the difficulty.
        """
        if self.shared_keys and self.shared_pct_right > 50:
            return f"{self.right_name} is largely contained in {self.left_name}: not an independent set"
        if not np.isfinite(self.auc_gap):
            # Saying "indistinguishable" here would turn "we could not measure it" into a finding.
            return "not measurable: too few molecules, or no feature varies; see the notes"

        close = np.isfinite(self.nn_ratio) and self.nn_ratio <= 1.0
        if self.auc_gap < 0.05:
            return f"indistinguishable: the probe does no better on the real pair than on {self.left_name} against itself"
        if close:
            return (
                f"separable but close: {self.right_name} is a distinguishable population, yet it sits no "
                f"farther from {self.left_name} than {self.left_name} sits from itself -- an analog-expansion "
                "regime, where scaffold- or cluster-held-out validation overstates the difficulty"
            )
        if self.auc_gap < 0.20:
            return "mildly shifted: a model trained on the left should mostly transfer"
        return "strongly shifted: the right set is a different chemical population, treat left-side validation as optimistic"

    def report_lines(self) -> list[str]:
        """The full terminal report."""
        left, right = self.left_name, self.right_name
        lines = [
            f"{left} ({self.left_rows} rows, {self.left_keys} skeletons) vs "
            f"{right} ({self.right_rows} rows, {self.right_keys} skeletons)",
            "",
            "overlap",
            f"  {self.shared_keys} shared skeletons "
            f"({self.shared_pct_left:.2f}% of {left}, {self.shared_pct_right:.2f}% of {right})",
            "",
            "separability",
            f"  probe AUC          {self.probe_auc:.3f}",
            f"  null AUC           {self.probe_auc_null:.3f}   ({left} split against itself)",
            f"  gap                {self.auc_gap:+.3f}",
            f"  descriptor AUC     {self.descriptor_auc:.3f}   (named features only)",
        ]
        for name, coefficient in self.probe_drivers:
            lines.append(f"      {name:<28} {coefficient:+.3f}")
        lines += [
            "",
            "distance",
            f"  median NN Tanimoto {self.nn_similarity_median:.3f}   ({right} to its nearest {left} molecule)",
            f"  nn_ratio           {self.nn_ratio:.3f}   (1.0 = {right} sits as close as {left} sits to itself)",
            f"  frac_out           {self.frac_out:.3f}   (0.05 = same distribution)",
            f"  MMD                {self.mmd:.4f}  p={self.mmd_p:.3f}",
            "",
            "coverage",
            f"  scaffolds of {right} present in {left}: {100 * self.scaffold_coverage:.1f}%",
            f"  clusters of {right} containing a {left} molecule: {100 * self.cluster_coverage:.1f}%",
            "",
            "what differs",
        ]
        lines += [f"  {s.describe()}" for s in self.feature_drivers] or ["  nothing separating"]
        if self.label_shifts:
            lines += ["", "labels"]
            lines += [f"  {s.describe(left, right, self.label_suffix)}" for s in self.label_shifts]
        lines += ["", f"verdict: {self.verdict()}"]
        lines += [f"note: {n}" for n in self.notes]
        return lines


def _keys(df: pl.DataFrame, smiles_column: str) -> set[str]:
    """Connectivity-block InChIKeys of a table's molecules."""
    from data_tools.standardize import connectivity_keys

    return {k for k in connectivity_keys(df[smiles_column].to_list()) if k}


def _scaffolds(smiles: list[str]) -> set[str]:
    """Bemis-Murcko scaffolds present in a list of molecules; unparseable ones contribute nothing."""
    from data_tools.split_column import _scaffold_of

    found = (_scaffold_of(smi, i) for i, smi in enumerate(smiles))
    return {s for s in found if s and not s.startswith("__invalid_")}


def _cluster_coverage(left_smiles: list[str], right_smiles: list[str], cutoff: float) -> float:
    """Share of the right set's molecules whose joint cluster also holds a left-set molecule.

    Measured on molecules rather than on clusters so that one huge covered cluster is not outvoted by a
    hundred covered singletons: this is "how much of B would a neighbour-based model find support for in A".
    """
    from data_tools.split_column import _butina_groups

    n_left = len(left_smiles)
    groups = _butina_groups(left_smiles + right_smiles, cutoff)
    covered = 0
    total = 0
    for key, members in groups.items():
        if key.startswith("__invalid_"):
            continue
        right_members = [m for m in members if m >= n_left]
        if not right_members:
            continue
        total += len(right_members)
        if any(m < n_left for m in members):
            covered += len(right_members)
    return covered / total if total else float("nan")


def _probe_drivers(
    X_left: np.ndarray,
    X_right: np.ndarray,
    names: list[str],
    *,
    top: int,
    seed: int,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """(AUC, top standardized coefficients) of a logistic probe on named features.

    Standardizing first is what makes the coefficients comparable: raw RDKit descriptors span molecular
    weight in the hundreds and ring counts in single digits, so unscaled coefficients rank by unit, not by
    importance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.vstack([X_left, X_right])
    keep = usable_columns(X)
    if keep.sum() == 0:
        return float("nan"), ()
    X = X[:, keep]
    kept_names = [n for n, k in zip(names, keep) if k]

    # Re-check the spread on the sliced matrix and divide by that same vector. Deciding "varies" from one
    # computation and dividing by another invites a zero denominator on a near-constant column.
    spread = X.std(axis=0)
    varies = spread > 0
    if not varies.any():
        return float("nan"), ()
    X = (X[:, varies] - X[:, varies].mean(axis=0)) / spread[varies]
    kept_names = [n for n, v in zip(kept_names, varies) if v]
    y = np.r_[np.zeros(len(X_left)), np.ones(len(X_right))]

    smallest = int(np.bincount(y.astype(int)).min())
    if smallest < 2:
        return float("nan"), ()
    folds = StratifiedKFold(n_splits=min(5, smallest), shuffle=True, random_state=seed)
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    proba = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))

    coefficients = model.fit(X, y).coef_[0]
    order = np.argsort(-np.abs(coefficients))[:top]
    return auc, tuple((kept_names[j], float(coefficients[j])) for j in order)


def compare(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_name: str = "left",
    right_name: str = "right",
    smiles_column: str = "SMILES",
    targets: list[str] | None = None,
    top_features: int = 6,
    cutoff: float = _DEFAULT_CUTOFF,
    max_samples: int = _MAX_PROBE,
    label_suffix: str = "",
    seed: int = 0,
) -> Comparison:
    """Measure every difference between two tables of molecules. Neither needs labels."""
    from scipy import stats

    from data_tools.describe import fingerprints
    from representations.embedding_metrics import (
        fingerprint_shift,
        mmd_rbf,
        standardize,
        subsample,
        two_sample_auc,
    )

    rng = np.random.default_rng(seed)
    notes: list[str] = []

    left_smiles, right_smiles = left[smiles_column].to_list(), right[smiles_column].to_list()
    left_keys, right_keys = _keys(left, smiles_column), _keys(right, smiles_column)

    # Binary, not counts: `tanimoto_nn_distances` is the bit form (intersection over union), so counts would
    # quietly stop being a Tanimoto. Binary is also what Butina clusters on, keeping every reading comparable.
    fp_left_all = fingerprints(left[smiles_column], counts=False)
    fp_right_all = fingerprints(right[smiles_column], counts=False)
    good_left = np.isfinite(fp_left_all).all(axis=1)
    good_right = np.isfinite(fp_right_all).all(axis=1)
    if not good_left.all() or not good_right.all():
        notes.append(
            f"dropped {int((~good_left).sum())} + {int((~good_right).sum())} unparseable molecules "
            "from the distance and probe readings"
        )
    fp_left, fp_right = fp_left_all[good_left], fp_right_all[good_right]

    rows_left = subsample(len(fp_left), max_samples, rng)
    rows_right = subsample(len(fp_right), max_samples, rng)
    sample_left, sample_right = fp_left[rows_left], fp_right[rows_right]

    # Headline probe, and the null it has to beat: a random split of the left set against itself.
    def probe(a: np.ndarray, b: np.ndarray, what: str) -> float:
        """Folded `two_sample_auc`, or NaN and a note when there is too little to cross-validate.

        Two guards, both about not turning an artefact into a finding.

        It stratifies into five folds, so fewer than five molecules on a side is not a low reading, it is
        no reading at all -- and reporting it as 0.5 would claim the two sets match.

        The result is folded about 0.5 because which side the classifier calls positive is arbitrary: 0.1 is
        as separable as 0.9. Unfolded, a set compared against one that shares its molecules scores near
        *zero* -- each held-out molecule's duplicate sits in the training fold under the opposite label, so
        the probe is taught to invert -- and that would read as "indistinguishable" exactly when the two
        sets are the same data.
        """
        if min(len(a), len(b)) < _MIN_PROBE:
            notes.append(f"{what}: needs {_MIN_PROBE} molecules a side, had {min(len(a), len(b))}")
            return float("nan")
        raw = two_sample_auc(a, b, seed)
        if raw < 0.35:
            notes.append(
                f"{what}: raw AUC {raw:.2f} is below chance, which means molecules appear on both sides; "
                "folded to its distance from 0.5"
            )
        return 0.5 + abs(raw - 0.5)

    probe_auc = probe(sample_left, sample_right, "probe AUC")
    half = rng.permutation(len(sample_left))
    cut = len(half) // 2
    probe_null = probe(sample_left[half[:cut]], sample_left[half[cut:]], "null AUC")

    shift, d_right = fingerprint_shift(sample_left, sample_right)
    nn_similarity = float(np.median(1.0 - d_right)) if len(d_right) else float("nan")

    # `standardize` drops every column that is constant across the left set. A congeneric enough left set
    # leaves nothing behind, and the distance machinery cannot run on a zero-width matrix.
    Z_left, Z_right = standardize(sample_left, sample_right)
    if Z_left.shape[1] == 0:
        notes.append(f"every fingerprint bit is constant across {left_name}, so MMD could not be computed")
        mmd, mmd_p = float("nan"), float("nan")
    else:
        mmd, mmd_p = mmd_rbf(Z_left, Z_right, seed)

    # Named features, on exactly the rows the probe and distance readings used, so every number in the
    # report describes the same molecules.
    left_rows = np.flatnonzero(good_left)[rows_left]
    right_rows = np.flatnonzero(good_right)[rows_right]
    X_left, feature_names = named_features(left[smiles_column].gather(left_rows.tolist()))
    X_right, _ = named_features(right[smiles_column].gather(right_rows.tolist()))
    X = np.vstack([X_left, X_right])
    membership = np.r_[np.zeros(len(X_left), dtype=bool), np.ones(len(X_right), dtype=bool)]
    drivers = feature_shifts(X, feature_names, membership, top=top_features)
    descriptor_auc, probe_drivers = _probe_drivers(X_left, X_right, feature_names, top=top_features, seed=seed)

    label_shifts = []
    for target in targets or []:
        if target not in left.columns or target not in right.columns:
            continue
        a = left[target].to_numpy().astype(np.float64)
        b = right[target].to_numpy().astype(np.float64)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        left_n, left_mean, left_std = target_stats(a)
        right_n, right_mean, right_std = target_stats(b)
        if left_n < 3 or right_n < 3:
            continue
        test = stats.ks_2samp(a, b)
        label_shifts.append(
            LabelShift(
                target, left_n, left_mean, left_std, right_n, right_mean, right_std,
                float(test.statistic), float(test.pvalue),
            )
        )

    left_scaffolds, right_scaffolds = _scaffolds(left_smiles), _scaffolds(right_smiles)
    scaffold_coverage = (
        len(left_scaffolds & right_scaffolds) / len(right_scaffolds) if right_scaffolds else float("nan")
    )

    return Comparison(
        left_name=left_name,
        right_name=right_name,
        left_rows=len(left),
        right_rows=len(right),
        left_keys=len(left_keys),
        right_keys=len(right_keys),
        shared_keys=len(left_keys & right_keys),
        probe_auc=probe_auc,
        probe_auc_null=probe_null,
        descriptor_auc=descriptor_auc,
        probe_drivers=probe_drivers,
        nn_similarity_median=nn_similarity,
        nn_ratio=shift["nn_ratio"],
        frac_out=shift["frac_out"],
        mmd=mmd,
        mmd_p=mmd_p,
        feature_drivers=tuple(drivers),
        label_shifts=tuple(label_shifts),
        scaffold_coverage=scaffold_coverage,
        cluster_coverage=_cluster_coverage(
            [left_smiles[i] for i in left_rows],
            [right_smiles[i] for i in right_rows],
            cutoff,
        ),
        label_suffix=label_suffix,
        notes=tuple(notes),
    )


def run(args) -> None:
    """`data compare` entry point."""
    from data_tools.describe import resolve_table, target_columns

    left, left_name = resolve_table(args.left, args.data_dir, smiles_column=args.smiles_column)
    right, right_name = resolve_table(args.right, args.data_dir, smiles_column=args.smiles_column)

    suffix = ""
    if args.targets is None:
        try:
            from data_tools.datasets import challenge

            spec = challenge()
            shared = [t for t in spec.targets if t in left.columns and t in right.columns]
            suffix = spec.label_suffix if shared else ""
        except ValueError:
            shared = []
        targets = shared or [t for t in target_columns(left) if t in right.columns]
    else:
        targets = target_columns(left, args.targets)

    comparison = compare(
        left,
        right,
        left_name=left_name,
        right_name=right_name,
        smiles_column=args.smiles_column,
        targets=targets,
        top_features=args.top_features,
        cutoff=args.cutoff,
        max_samples=args.max_samples,
        label_suffix=suffix,
        seed=args.seed,
    )
    print("\n".join(comparison.report_lines()))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(comparison.report_lines()) + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
