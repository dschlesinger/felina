"""Clustering, cluster summaries, and the figure `data explore` writes."""

import numpy as np
import polars as pl
import pytest

from data_tools.dataset_explore import draw, explore
from data_tools.describe import feature_shifts, named_features, target_columns, target_stats

# Two chemotypes that share nothing: long aliphatic chains, and fused aromatics carrying a basic amine.
_CHAINS = [
    "CCCCCCCCCO",
    "CCCCCCCCCCO",
    "CCCCCCCCCCCO",
    "CCCCCCCCCCCCO",
    "CCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCO",
]
# A congeneric series: one naphthalene scaffold, one substituent position, six substituents. Butina holds it
# together at the default cutoff, which is what makes the expected cluster membership here deterministic.
_AROMATICS = [
    "CN(C)CCc1ccc2ccccc2c1C",
    "CN(C)CCc1ccc2ccccc2c1CC",
    "CN(C)CCc1ccc2ccccc2c1F",
    "CN(C)CCc1ccc2ccccc2c1Cl",
    "CN(C)CCc1ccc2ccccc2c1Br",
    "CN(C)CCc1ccc2ccccc2c1OC",
]


@pytest.fixture
def two_chemotypes() -> pl.DataFrame:
    """Six chains labelled low, six aromatics labelled high, so clusters and labels line up."""
    return pl.DataFrame({
        "SMILES": _CHAINS + _AROMATICS,
        "Molecule_Name": [f"M{i}" for i in range(len(_CHAINS) + len(_AROMATICS))],
        "CYP3A4_pIC50_direct_inhibition": [4.0, 4.1, 4.2, 3.9, 4.0, 4.1] + [7.0, 7.1, 6.9, 7.2, 7.0, 6.8],
    })


def test_distinct_chemotypes_land_in_separate_clusters(two_chemotypes):
    """The point of the command: two chemotypes must not be reported as one blob."""
    result = explore(two_chemotypes, min_size=3, targets=["CYP3A4_pIC50_direct_inhibition"])

    assert len(result.clusters) == 2
    memberships = sorted((set(c.members) for c in result.clusters), key=min)
    assert memberships == [set(range(6)), set(range(6, 12))]


def test_cluster_carries_its_targets_size_and_medoid(two_chemotypes):
    result = explore(two_chemotypes, min_size=3, targets=["CYP3A4_pIC50_direct_inhibition"])

    for cluster in result.clusters:
        assert cluster.size == 6
        assert cluster.medoid in cluster.members
        assert cluster.medoid_smiles == two_chemotypes["SMILES"][cluster.medoid]
        summary = cluster.targets[0]
        assert summary.n == 6
        # Each cluster's label mean sits with its own chemotype, not with the dataset's overall mean of 5.5.
        assert summary.mean == pytest.approx(4.05, abs=0.2) or summary.mean == pytest.approx(7.0, abs=0.2)
        assert summary.std < 0.5


def test_standout_features_name_what_makes_a_cluster_itself(two_chemotypes):
    """A reported feature has to be one a chemist would recognize as the difference."""
    result = explore(two_chemotypes, min_size=3, top_features=6, targets=[])

    aromatic = next(c for c in result.clusters if "c1ccc2ccccc2c1" in c.medoid_smiles)
    named = {s.name for s in aromatic.standout}
    assert any("Aromatic" in n or "aromatic" in n for n in named), named
    # And the direction is right: the aromatic cluster sits higher on its aromatic features.
    for shift in aromatic.standout:
        if "Aromatic" in shift.name:
            assert shift.d > 0
            assert shift.group_mean > shift.rest_mean


def test_clusters_below_the_floor_are_counted_not_summarized(two_chemotypes):
    """A mean over two molecules is noise; it must be excluded but still accounted for."""
    with_singleton = pl.concat([
        two_chemotypes,
        pl.DataFrame({
            "SMILES": ["[Na+].[Cl-]"],
            "Molecule_Name": ["salt"],
            "CYP3A4_pIC50_direct_inhibition": [5.0],
        }),
    ])
    result = explore(with_singleton, min_size=3, targets=["CYP3A4_pIC50_direct_inhibition"])

    assert len(result.clusters) == 2
    assert result.small_clusters == 1
    assert result.small_molecules == 1
    assert 13 not in [m for c in result.clusters for m in c.members]


def test_max_clusters_caps_the_report_but_not_the_count(two_chemotypes):
    result = explore(two_chemotypes, min_size=3, max_clusters=1, targets=[])

    assert len(result.clusters) == 1
    assert result.clusters[0].rank == 1
    assert result.clusters[0].size == 6


def test_clusters_are_ordered_largest_first():
    df = pl.DataFrame({"SMILES": _CHAINS + _AROMATICS[:3]})
    result = explore(df, min_size=3, targets=[])

    sizes = [c.size for c in result.clusters]
    assert sizes == sorted(sizes, reverse=True)
    assert [c.rank for c in result.clusters] == list(range(1, len(sizes) + 1))


def test_medoid_is_the_member_nearest_the_others(two_chemotypes):
    """A medoid on the cluster's edge would make the panel misrepresent the cluster."""
    from data_tools.describe import fingerprints
    from data_tools.dataset_explore import _tanimoto_matrix

    result = explore(two_chemotypes, min_size=3, targets=[])
    fp = fingerprints(two_chemotypes["SMILES"])

    for cluster in result.clusters:
        members = np.asarray(cluster.members)
        sims = _tanimoto_matrix(fp[members], fp[members])
        np.fill_diagonal(sims, 0.0)
        assert members[int(np.argmax(sims.mean(axis=1)))] == cluster.medoid


def test_unparseable_molecules_do_not_become_clusters():
    df = pl.DataFrame({"SMILES": _CHAINS + ["not_a_smiles", "also)(bad"]})
    result = explore(df, min_size=3, targets=[])

    assert len(result.clusters) == 1
    assert result.n_clustered == len(_CHAINS)
    assert all(m < len(_CHAINS) for c in result.clusters for m in c.members)


def test_draw_writes_a_real_png(two_chemotypes, tmp_path):
    result = explore(two_chemotypes, min_size=3, targets=["CYP3A4_pIC50_direct_inhibition"])
    out = draw(result, tmp_path / "nested" / "clusters.png")

    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 10_000


def test_draw_refuses_an_empty_exploration_instead_of_writing_a_blank_figure(two_chemotypes, tmp_path):
    result = explore(two_chemotypes, min_size=99, targets=[])

    assert result.clusters == ()
    with pytest.raises(ValueError, match="nothing to draw"):
        draw(result, tmp_path / "empty.png")


def test_report_lines_mention_every_cluster_and_its_labels(two_chemotypes):
    result = explore(two_chemotypes, min_size=3, targets=["CYP3A4_pIC50_direct_inhibition"])
    text = "\n".join(result.report_lines())

    for cluster in result.clusters:
        assert f"#{cluster.rank}" in text
        assert cluster.medoid_smiles in text
    assert "CYP3A4" in text


def test_target_stats_refuses_to_summarize_two_molecules():
    n, mean, std = target_stats(np.array([1.0, 2.0]))
    assert n == 2
    assert np.isnan(mean) and np.isnan(std)


def test_target_stats_ignores_missing_labels():
    n, mean, std = target_stats(np.array([1.0, np.nan, 3.0, 2.0]))
    assert n == 3
    assert mean == pytest.approx(2.0)


def test_target_columns_skips_identifiers_and_keeps_labels(two_chemotypes):
    found = target_columns(two_chemotypes)
    assert found == ["CYP3A4_pIC50_direct_inhibition"]


def test_target_columns_names_the_missing_column():
    with pytest.raises(ValueError, match="nope"):
        target_columns(pl.DataFrame({"SMILES": ["C"]}), ["nope"])


def test_feature_shifts_returns_nothing_when_a_side_is_too_small():
    X, names = named_features(pl.Series(_CHAINS))
    mask = np.zeros(len(_CHAINS), dtype=bool)
    mask[0] = True
    assert feature_shifts(X, names, mask) == []
