"""Overlap, separability, distance and label-shift readings from `data compare`.

The load-bearing test here is `test_probe_null_is_near_chance_when_a_set_is_split_against_itself`: the probe
AUC is only readable against its own null, and if the null were broken every other number would be too.
"""

import numpy as np
import polars as pl
import pytest

from data_tools.dataset_compare import _cluster_coverage, _scaffolds, compare

_CHAINS = [
    "CCCCCCCCCO",
    "CCCCCCCCCCO",
    "CCCCCCCCCCCO",
    "CCCCCCCCCCCCO",
    "CCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCO",
    "CCCCCCCCCCCCCCCCCCCCO",
]
_AMINES = [
    "CN(C)CCc1ccc2ccccc2c1",
    "CN(C)CCc1ccc2ccccc2c1C",
    "CN(C)CCc1ccc2ccccc2c1CC",
    "CN(C)CCc1cc(C)c2ccccc2c1",
    "CN(C)CCc1ccc2cc(C)ccc2c1",
    "CN(C)CCc1ccc2ccc(C)cc2c1",
    "CCN(C)CCc1ccc2ccccc2c1",
    "CN(CC)CCc1ccc2ccccc2c1",
    "CN(C)CCCc1ccc2ccccc2c1",
    "CN(C)CCc1ccc2ccccc2c1F",
    "CN(C)CCc1ccc2ccccc2c1Cl",
    "CN(C)CCc1ccc2ccccc2c1Br",
]


def _frame(smiles: list[str], label: float | None = None) -> pl.DataFrame:
    data: dict[str, list] = {"SMILES": smiles}
    if label is not None:
        data["CYP3A4_pIC50_direct_inhibition"] = [label + 0.1 * i for i in range(len(smiles))]
    return pl.DataFrame(data)


def test_probe_null_is_near_chance_when_a_set_is_split_against_itself():
    """The calibration the headline AUC is meaningless without.

    A probe handed two random halves of one dataset has nothing to learn, so its AUC must sit near 0.5. If
    this drifts high, the null is inflated by dimensionality and every raw AUC in the report is overstated.
    """
    half = len(_CHAINS + _AMINES) // 2
    pool = _CHAINS + _AMINES
    result = compare(_frame(pool[:half]), _frame(pool[half:]), seed=0)

    assert result.probe_auc_null == pytest.approx(0.5, abs=0.35)
    assert np.isfinite(result.probe_auc_null)


def test_disjoint_chemotypes_separate_far_above_the_null():
    result = compare(_frame(_CHAINS), _frame(_AMINES), left_name="chains", right_name="amines")

    assert result.probe_auc > 0.9
    assert result.auc_gap > 0.2
    assert "shifted" in result.verdict()


def test_the_same_molecules_compared_to_themselves_show_no_gap():
    result = compare(_frame(_CHAINS + _AMINES), _frame(_CHAINS + _AMINES))

    assert result.auc_gap == pytest.approx(0.0, abs=0.25)
    assert result.shared_keys == result.left_keys == result.right_keys
    assert result.nn_similarity_median == pytest.approx(1.0, abs=1e-6)
    assert result.scaffold_coverage == pytest.approx(1.0)
    assert result.cluster_coverage == pytest.approx(1.0)


def test_overlap_counts_salt_and_stereo_variants_as_the_same_skeleton():
    """Raw SMILES comparison would call these four different molecules and miss the leak."""
    left = _frame(["CC(=O)Oc1ccccc1C(=O)O", "CN(C)CCc1ccc2ccccc2c1"])
    right = _frame(["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "C[C@H](N)Cc1ccccc1"])

    result = compare(left, right)
    assert result.shared_keys == 1
    assert result.shared_pct_right == pytest.approx(50.0)


def test_a_contained_set_is_called_out_as_not_independent():
    result = compare(_frame(_CHAINS + _AMINES), _frame(_AMINES))
    assert "not an independent set" in result.verdict()


def test_nearest_neighbour_similarity_falls_as_the_sets_diverge():
    """The reading that separates 'novel scaffolds' from 'analogs of what we trained on'."""
    near = compare(_frame(_AMINES[:6]), _frame(_AMINES[6:]))
    far = compare(_frame(_CHAINS), _frame(_AMINES))

    assert near.nn_similarity_median > far.nn_similarity_median
    assert far.nn_ratio > near.nn_ratio


def test_feature_drivers_name_the_axis_the_sets_differ_on():
    result = compare(_frame(_CHAINS), _frame(_AMINES), top_features=10)

    names = {s.name for s in result.feature_drivers}
    assert any("Aromatic" in n or "Ring" in n for n in names), names
    # The amine set is the right-hand one, so its aromatic features must read positive.
    for shift in result.feature_drivers:
        if "NumAromaticRings" == shift.name:
            assert shift.d > 0


def test_descriptor_probe_reports_an_auc_and_its_coefficients():
    result = compare(_frame(_CHAINS), _frame(_AMINES), top_features=4)

    assert result.descriptor_auc > 0.9
    assert len(result.probe_drivers) == 4
    assert all(isinstance(name, str) and np.isfinite(value) for name, value in result.probe_drivers)


def test_label_shift_is_detected_when_the_chemistry_matches():
    """Same molecules, different assay scale: a shift no fingerprint reading would catch."""
    left = _frame(_AMINES, label=4.0)
    right = _frame(_AMINES, label=7.0)

    result = compare(left, right, targets=["CYP3A4_pIC50_direct_inhibition"])

    assert result.auc_gap == pytest.approx(0.0, abs=0.3)  # chemically identical
    shift = result.label_shifts[0]
    assert shift.left_mean == pytest.approx(4.55, abs=0.1)
    assert shift.right_mean == pytest.approx(7.55, abs=0.1)
    assert shift.ks == pytest.approx(1.0)
    assert shift.p < 0.01


def test_matching_labels_are_not_flagged():
    result = compare(
        _frame(_AMINES, label=5.0),
        _frame(_AMINES, label=5.0),
        targets=["CYP3A4_pIC50_direct_inhibition"],
    )
    assert result.label_shifts[0].p > 0.05


def test_a_target_missing_from_one_side_is_skipped_not_fabricated():
    result = compare(
        _frame(_AMINES, label=5.0),
        _frame(_AMINES),
        targets=["CYP3A4_pIC50_direct_inhibition"],
    )
    assert result.label_shifts == ()


def test_cluster_coverage_is_zero_for_chemically_disjoint_sets():
    assert _cluster_coverage(_CHAINS, _AMINES, 0.4) == pytest.approx(0.0)
    assert _cluster_coverage(_CHAINS, _CHAINS, 0.4) == pytest.approx(1.0)


def test_scaffold_coverage_ignores_unparseable_molecules():
    assert _scaffolds(["not_a_smiles"]) == set()
    assert len(_scaffolds(_AMINES)) >= 1


def test_unparseable_molecules_are_dropped_and_reported():
    result = compare(_frame(_CHAINS + ["not_a_smiles"]), _frame(_AMINES))

    assert any("unparseable" in note for note in result.notes)
    assert result.left_rows == len(_CHAINS) + 1  # the row count is still the truth about the file


def test_report_lines_carry_every_heading():
    result = compare(_frame(_CHAINS, label=4.0), _frame(_AMINES, label=6.0),
                     left_name="chains", right_name="amines",
                     targets=["CYP3A4_pIC50_direct_inhibition"])
    text = "\n".join(result.report_lines())

    for heading in ("overlap", "separability", "distance", "coverage", "what differs", "labels", "verdict"):
        assert heading in text
    assert "null AUC" in text
    assert "chains" in text and "amines" in text


def test_comparison_is_reproducible_under_a_seed():
    left, right = _frame(_CHAINS), _frame(_AMINES)
    first = compare(left, right, seed=7)
    second = compare(left, right, seed=7)

    assert first.probe_auc == second.probe_auc
    assert first.probe_auc_null == second.probe_auc_null
    assert [s.name for s in first.feature_drivers] == [s.name for s in second.feature_drivers]
