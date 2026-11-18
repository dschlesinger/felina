"""`data explore`: cluster a dataset and draw what each cluster actually contains.

A list of cluster sizes says nothing about chemistry. What is worth knowing before modelling is *which*
chemotypes a table is made of, how big each one is, how its labels sit relative to the rest, and what makes
it the cluster it is. So each cluster is reported as four things at once:

  the medoid         the molecule nearest every other member, drawn -- the cluster's face
  the size           members, and the share of the dataset they are
  the labels         mean and standard deviation per target over the members that carry one
  the standout       the named features that separate this cluster from the rest of the dataset,
                     with an effect size, so the panel says *why* these molecules group

Clustering is Butina on Morgan fingerprints, the same algorithm `split_column` uses for grouped splits, so a
cluster seen here is the same unit a cluster-held-out CV fold would move around.
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.describe import FeatureShift, feature_shifts, named_features, target_stats

_DEFAULT_CUTOFF = 0.4  # Butina *distance* cutoff: members sit within Tanimoto 0.6 of a centroid
_MEDOID_CAP = 400  # members sampled when finding a big cluster's medoid; the medoid barely moves above this
_PANEL_COLUMNS = 4


@dataclass(frozen=True)
class TargetSummary:
    """One target's distribution inside one cluster."""

    name: str
    n: int
    mean: float
    std: float

    def describe(self, suffix: str = "") -> str:
        label = self.name.removesuffix(suffix) if suffix else self.name
        if not math.isfinite(self.mean):
            return f"{label}: n={self.n}"
        return f"{label}: {self.mean:.2f} +/- {self.std:.2f} (n={self.n})"


@dataclass(frozen=True)
class Cluster:
    """One Butina cluster: who is in it, its face, its labels, and what marks it out."""

    rank: int  # 1 = largest
    members: tuple[int, ...]  # row indices into the explored frame
    medoid: int  # row index of the member nearest all the others
    medoid_smiles: str
    targets: tuple[TargetSummary, ...]
    standout: tuple[FeatureShift, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Exploration:
    """Every cluster of one dataset, largest first, plus what was left over."""

    name: str
    n_rows: int
    n_clustered: int
    cutoff: float
    clusters: tuple[Cluster, ...]
    small_clusters: int  # clusters dropped for being under min_size
    small_molecules: int  # molecules in them
    targets: tuple[str, ...]
    label_suffix: str = ""

    def report_lines(self) -> list[str]:
        """A terminal-readable summary, one block per cluster."""
        lines = [
            f"{self.name}: {self.n_rows} molecules, {self.n_clustered} clustered at Butina cutoff {self.cutoff}",
            f"{len(self.clusters)} clusters at or above the size floor; "
            f"{self.small_clusters} smaller ones hold {self.small_molecules} molecules",
            "",
        ]
        for cluster in self.clusters:
            share = 100.0 * cluster.size / self.n_rows if self.n_rows else 0.0
            lines.append(f"#{cluster.rank}  n={cluster.size} ({share:.1f}%)  medoid {cluster.medoid_smiles}")
            for target in cluster.targets:
                lines.append(f"    {target.describe(self.label_suffix)}")
            for shift in cluster.standout:
                lines.append(f"    {shift.describe()}")
            lines.append("")
        return lines


def _tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between two count-fingerprint blocks, via the min/max form.

    The min/max (Ruzicka) form reduces to the usual bit Tanimoto when the inputs are 0/1, so one code path
    serves both; it is what makes counts usable as a similarity at all.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    # Broadcasting the full (n, m, bits) stack would be enormous, so accumulate per row.
    out = np.empty((len(a), len(b)), dtype=np.float32)
    for i, row in enumerate(a):
        minimum = np.minimum(row, b).sum(axis=1)
        maximum = np.maximum(row, b).sum(axis=1)
        out[i] = np.where(maximum > 0, minimum / np.maximum(maximum, 1e-9), 0.0)
    return out


def _medoid(fp: np.ndarray, members: np.ndarray, rng: np.random.Generator) -> int:
    """The member with the highest mean similarity to the others; its own row index."""
    if len(members) == 1:
        return int(members[0])
    sample = members
    if len(members) > _MEDOID_CAP:
        sample = rng.choice(members, size=_MEDOID_CAP, replace=False)
    sims = _tanimoto_matrix(fp[sample], fp[sample])
    np.fill_diagonal(sims, 0.0)
    return int(sample[int(np.argmax(sims.mean(axis=1)))])


def explore(
    df: pl.DataFrame,
    *,
    name: str = "dataset",
    smiles_column: str = "SMILES",
    targets: list[str] | None = None,
    cutoff: float = _DEFAULT_CUTOFF,
    min_size: int = 5,
    top_features: int = 3,
    max_clusters: int = 12,
    label_suffix: str = "",
    cache_dir: Path | None = None,
    seed: int = 0,
) -> Exploration:
    """Cluster `df` and summarize every cluster at or above `min_size`, largest first.

    `max_clusters` caps how many are summarized in full, since the expensive part is the per-cluster feature
    comparison and nobody reads the 400th cluster of a long tail. Clusters below `min_size` are counted but
    not summarized: a mean over four molecules is noise with a decimal point.
    """
    from data_tools.describe import fingerprints
    from data_tools.split_column import _butina_groups

    rng = np.random.default_rng(seed)
    smiles = df[smiles_column].to_list()
    groups = _butina_groups(smiles, cutoff)
    # `_butina_groups` gives each unparseable molecule a singleton group of its own; those are not clusters.
    clustered = {k: v for k, v in groups.items() if not k.startswith("__invalid_")}
    n_clustered = sum(len(v) for v in clustered.values())

    ordered = sorted(clustered.values(), key=len, reverse=True)
    big = [m for m in ordered if len(m) >= min_size]
    small = [m for m in ordered if len(m) < min_size]

    fp = fingerprints(df[smiles_column], cache_dir=cache_dir)
    X, feature_names = named_features(df[smiles_column], cache_dir=cache_dir)
    target_names = targets if targets is not None else []

    clusters: list[Cluster] = []
    for rank, members in enumerate(big[:max_clusters], start=1):
        members_array = np.asarray(members, dtype=int)
        mask = np.zeros(len(df), dtype=bool)
        mask[members_array] = True

        summaries = []
        for target in target_names:
            values = df[target].to_numpy().astype(np.float64)[members_array]
            n, mean, std = target_stats(values)
            summaries.append(TargetSummary(target, n, mean, std))

        # Once, not once per field: `_medoid` draws from `rng` for large clusters, so a second call could
        # sample differently and hand back an index that disagrees with the SMILES stored beside it.
        medoid = _medoid(fp, members_array, rng)
        clusters.append(
            Cluster(
                rank=rank,
                members=tuple(int(i) for i in members_array),
                medoid=medoid,
                medoid_smiles=smiles[medoid],
                targets=tuple(summaries),
                standout=tuple(feature_shifts(X, feature_names, mask, top=top_features)),
            )
        )

    return Exploration(
        name=name,
        n_rows=len(df),
        n_clustered=n_clustered,
        cutoff=cutoff,
        clusters=tuple(clusters),
        small_clusters=len(small),
        small_molecules=sum(len(m) for m in small),
        targets=tuple(target_names),
        label_suffix=label_suffix,
    )


def _molecule_png(smiles: str, size: tuple[int, int] = (420, 320)) -> bytes | None:
    """Render one molecule to PNG bytes, or None if RDKit cannot parse it."""
    from rdkit import Chem  # type: ignore[import]
    from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore[import]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw(exploration: Exploration, out_path: Path) -> Path:
    """Write the cluster figure: one panel per cluster, medoid drawn, caption underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clusters = exploration.clusters
    if not clusters:
        raise ValueError(
            f"{exploration.name}: no cluster reached the size floor, so there is nothing to draw "
            "(lower --min-size or raise --cutoff)"
        )

    columns = min(_PANEL_COLUMNS, len(clusters))
    rows = math.ceil(len(clusters) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 5.0 * rows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for panel, cluster in enumerate(clusters):
        ax = axes[panel // columns][panel % columns]
        png = _molecule_png(cluster.medoid_smiles)
        if png is not None:
            ax.imshow(plt.imread(io.BytesIO(png), format="png"))
        else:
            ax.text(0.5, 0.5, "unparseable medoid", ha="center", va="center", fontsize=9)

        share = 100.0 * cluster.size / exploration.n_rows if exploration.n_rows else 0.0
        ax.set_title(f"#{cluster.rank}   n={cluster.size}  ({share:.1f}%)", fontsize=11, fontweight="bold")

        caption = [t.describe(exploration.label_suffix) for t in cluster.targets]
        if caption:
            caption.append("")
        caption += [
            f"{'+' if s.d > 0 else '-'} {s.name}  {s.group_mean:.3g} vs {s.rest_mean:.3g}  (d={s.d:+.2f})"
            for s in cluster.standout
        ]
        ax.text(
            0.0,
            -0.04,
            "\n".join(caption) or "no separating feature",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
        )

    tail = (
        f"; {exploration.small_clusters} clusters under the size floor hold {exploration.small_molecules}"
        if exploration.small_clusters
        else ""
    )
    figure.suptitle(
        f"{exploration.name} — {exploration.n_rows} molecules, Butina cutoff {exploration.cutoff}, "
        f"showing {len(clusters)} clusters{tail}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def run(args) -> None:
    """`data explore` entry point."""
    from data_tools.describe import resolve_table, target_columns

    df, name = resolve_table(args.dataset, args.data_dir, smiles_column=args.smiles_column)
    if args.max_molecules and len(df) > args.max_molecules:
        print(f"sampling {args.max_molecules} of {len(df)} molecules", file=sys.stderr)
        df = df.sample(n=args.max_molecules, shuffle=True, seed=args.seed)

    suffix = ""
    if args.targets is None:
        try:
            from data_tools.datasets import challenge

            spec = challenge()
            wanted = [t for t in spec.targets if t in df.columns]
            suffix = spec.label_suffix if wanted else ""
        except ValueError:
            wanted = []
        targets = wanted or target_columns(df)
    else:
        targets = target_columns(df, args.targets)

    exploration = explore(
        df,
        name=name,
        smiles_column=args.smiles_column,
        targets=targets,
        cutoff=args.cutoff,
        min_size=args.min_size,
        top_features=args.top_features,
        max_clusters=args.max_clusters,
        label_suffix=suffix,
        cache_dir=args.feature_dir,
        seed=args.seed,
    )
    print("\n".join(exploration.report_lines()))

    out = Path(args.out) if args.out else Path("reports") / f"{name}_clusters.png"
    print(f"wrote {draw(exploration, out)}", file=sys.stderr)
