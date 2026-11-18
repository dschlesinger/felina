"""Overlap, separability, distance and label-shift readings from `data compare`.

The load-bearing test here is `test_probe_null_is_near_chance_when_a_set_is_split_against_itself`: the probe
AUC is only readable against its own null, and if the null were broken every other number would be too.
"""

import numpy as np
import polars as pl
import pytest

from data_tools.dataset_compare import _cluster_coverage, _scaffolds, compare

# Aliphatics, varied enough that Morgan has bits to work with. A pure homologous series would not be: every
# radius-2 environment in CCCCCCCCCO is also in CCCCCCCCCCCO, so the two have identical fingerprints and
# every bit is constant across the set. `test_degenerate_input_reports_nothing_rather_than_crashing`
# covers that case on purpose.
_CHAINS = [
    "CCCCCCCCCO",
    "CCCCCCCC(C)O",
    "CCCCCC(O)CCC",
    "OCCCCCCCCCN",
    "CCCCCCCCC(=O)O",
    "CCCCCCCCCC(=O)OC",
    "CC(C)CCCCCCO",
    "OCC(O)CCCCCCC",
    "CCCCCCCCS",
    "CCCCCCCCCBr",
    "CC(C)(C)CCCCCO",
    "NCCCCCCCCCC",
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


def _many_chains() -> list[str]:
    """~50 aliphatics with distinct fingerprints.

    Length alone is not enough to vary: every radius-2 environment in CCCCCCO also occurs in CCCCCCCCO, so
    a homologous series has one fingerprint repeated and no bit to separate anything on. The branch has to
    move along the chain for the environments themselves to differ.
    """
    tails = ["O", "N", "S", "C(=O)O", "C(=O)OC", "Br", "C#N"]
    out = []
    for tail in tails:
        for position in range(1, 7):
            body = "C" * position + "(C)" + "C" * (7 - position)
            out.append(f"C{body}{tail}")
    out += [f"OC{'C' * n}(C)CO" for n in (2, 3, 4, 5)]
    return sorted(set(out))


def _many_amines() -> list[str]:
    """~50 distinct aromatic amines: one scaffold family, varied substituent and linker."""
    subs = ["", "C", "CC", "F", "Cl", "Br", "OC", "C#N", "C(F)(F)F"]
    linkers = ["CC", "CCC", "CCCC"]
    heads = ["CN(C)", "CCN(C)", "CN(CC)"]
    out = []
    for head in heads:
        for linker in linkers:
            for sub in subs[:6]:
                out.append(f"{head}{linker}c1ccc2ccccc2c1{sub}")
    return sorted(set(out))


# Six mutually unrelated scaffolds, each with a substituent slot. Mutual distinctness is the point: when
# parents resemble each other, an analog's nearest neighbour is legitimately somebody else's parent, and the
# image collapses below the number of series for a real reason rather than a bug.
_SERIES = (
    "CN(C)CCc1ccc2ccccc2c1{}",
    "O=C(Nc1ccccc1{})c1ccccc1",
    "O=S(=O)(Nc1ccccc1{})c1cccs1",
    "N#CCNC(=O)c1csc(-c2ccc3OCOc3c2{})n1",
    "COc1ccc(-c2noc(CCC(=O)N3CCCCC3)n2)cc1{}",
    "O=C(NCCc1nc(-c2ccccc2{})no1)c1ccccc1",
)
_SERIES_SUBS = ("C", "CC", "F", "Cl")


def _parents() -> list[str]:
    return [template.format("") for template in _SERIES]


def _analogs() -> list[str]:
    return [template.format(sub) for template in _SERIES for sub in _SERIES_SUBS]


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
    """At a realistic sample size, two unrelated chemotypes must clear the null decisively."""
    result = compare(_frame(_many_chains()), _frame(_many_amines()), left_name="chains", right_name="amines")

    assert result.probe_auc > 0.9
    assert result.auc_gap_sigma > 3.0
    assert result.separable
    assert result.nn_ratio > 1.0  # and they really are far apart, not merely distinguishable
    assert "strongly shifted" in result.verdict()


def test_a_dozen_molecules_cannot_establish_separability_even_at_a_perfect_auc():
    """The reading the null exists to prevent.

    Twelve molecules a side separate perfectly -- AUC 1.0 -- while random halves of the left set alone
    score 0.66 and scatter by 0.17. One in twenty such splits would look near-perfect by chance, so a
    perfect score here is two standard deviations of noise, not evidence, and the verdict must say so.
    """
    result = compare(_frame(_CHAINS), _frame(_AMINES))

    assert result.probe_auc > 0.9  # perfectly separated...
    assert result.probe_auc_null_sd > 0.1  # ...but the null scatters nearly as far
    assert result.auc_gap_sigma < 3.0
    assert not result.separable
    assert "indistinguishable" in result.verdict()


def test_a_separable_but_nearby_set_is_not_called_strongly_shifted():
    """The reading one axis alone gets wrong.

    A congeneric expansion around molecules already in the left set is distinguishable as a population --
    the probe finds it easily -- while sitting closer to the left set than the left set sits to itself.
    Calling that "strongly shifted" would point at exactly the held-out split that overstates difficulty.
    """
    # A diverse left set, and a right set of molecules absent from it but analogous to part of it.
    result = compare(
        _frame(_AMINES[:6] + _CHAINS), _frame(_AMINES[6:]), left_name="train", right_name="expansion"
    )

    assert result.shared_keys == 0  # genuinely held out, not merely contained
    assert result.auc_gap > 0.2  # and genuinely separable
    assert result.nn_ratio <= 1.0
    assert "analog-expansion" in result.verdict()
    assert "strongly shifted" not in result.verdict()


def test_the_same_molecules_compared_to_themselves_are_called_identical():
    """Every chemistry reading must say "the same", and the verdict must lead with the containment."""
    result = compare(_frame(_CHAINS + _AMINES), _frame(_CHAINS + _AMINES))

    assert result.shared_keys == result.left_keys == result.right_keys
    assert result.nn_similarity_median == pytest.approx(1.0, abs=1e-6)
    assert result.scaffold_coverage == pytest.approx(1.0)
    assert result.cluster_coverage == pytest.approx(1.0)
    assert "not an independent set" in result.verdict()


def test_duplicated_molecules_across_sides_are_called_out_not_read_as_agreement():
    """The probe is taught to invert when a held-out molecule's twin sits in the training fold.

    Unfolded that lands near AUC 0, which would read as the two sets matching -- the opposite of what a
    perfectly inverted probe means. Folded it reads as separable, and a note names the cause.
    """
    result = compare(_frame(_CHAINS + _AMINES), _frame(_CHAINS + _AMINES))

    assert result.probe_auc >= 0.5
    assert any("below chance" in note for note in result.notes)


def test_overlap_counts_salt_and_stereo_variants_as_the_same_skeleton():
    """Raw SMILES comparison would call these four different molecules and miss the leak."""
    left = _frame(["CC(=O)Oc1ccccc1C(=O)O", "CN(C)CCc1ccc2ccccc2c1"])
    right = _frame(["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "C[C@H](N)Cc1ccccc1"])

    result = compare(left, right)
    assert result.shared_keys == 1
    assert result.shared_pct_right == pytest.approx(50.0)


def test_exact_overlap_separates_stereoisomers_but_strips_salts():
    """The full InChIKey keeps stereo, so enantiomers share a skeleton but are not an exact match."""
    left = _frame(["CC(=O)Oc1ccccc1C(=O)O", "C[C@@H](N)Cc1ccccc1"])
    right = _frame(["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "C[C@H](N)Cc1ccccc1"])

    result = compare(left, right)
    assert result.shared_keys == 2
    assert result.shared_exact == 1  # the salt strips to aspirin; the enantiomers stay apart
    assert result.exact_pct_right == pytest.approx(50.0)
    assert any("1 exact matches" in line for line in result.report_lines())


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
    assert {"iso_aromatic_frac", "FractionCSP3"} & names, names
    by_name = {s.name: s for s in result.feature_drivers}
    # The amine set is the right-hand one: more aromatic, less saturated. Both directions must agree.
    assert by_name["iso_aromatic_frac"].d > 0
    assert by_name["FractionCSP3"].d < 0
    # A cleanly separating feature legitimately earns a large d when within-group spread is small, so the
    # bound here only has to catch the pathology: a near-constant descriptor divided by its rounding error,
    # which reached 7e15 and outranked every real difference before the scale guard.
    assert all(np.isfinite(s.d) and abs(s.d) < 1e6 for s in result.feature_drivers), [
        (s.name, s.d) for s in result.feature_drivers
    ]


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

    # Chemically the same set; only the labels moved, which no fingerprint reading could see.
    assert result.shared_keys == result.left_keys
    assert result.nn_similarity_median == pytest.approx(1.0, abs=1e-6)
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


def test_small_sets_report_no_probe_rather_than_a_misleading_one():
    """Three molecules cannot be cross-validated. The answer is 'unmeasured', not 0.5."""
    result = compare(_frame(_AMINES[:3]), _frame(_CHAINS[:3]))

    assert np.isnan(result.probe_auc)
    assert any("a side" in note for note in result.notes)
    assert "not measurable" in result.verdict()


def test_degenerate_input_reports_nothing_rather_than_crashing():
    """A homologous series has one fingerprint repeated, so there is no bit to separate anything on."""
    series = [f"{'C' * n}O" for n in range(9, 21)]
    result = compare(_frame(series), _frame(series))

    assert np.isnan(result.mmd)
    assert any("constant" in note for note in result.notes)


def test_neighbour_image_counts_the_parents_an_expansion_collapses_onto():
    """The reading that exposes an analog expansion.

    Six parents, each with four analogs. The 24 analogs are 24 rows, but as far as the parent set is
    concerned they are six molecules wearing different substituents, and the image count must say six
    rather than flattering the expansion with its row count.
    """
    result = compare(_frame(_parents()), _frame(_analogs()), left_name="parents", right_name="analogs")

    image = result.image_right_in_left
    assert image.source_n == 24
    assert image.image_n == 6  # six parents account for all 24
    assert image.fan_mean == pytest.approx(4.0)
    assert image.fan_max == 4  # evenly spread: a congeneric expansion, not one promiscuous match


def test_neighbour_image_of_unrelated_sets_is_not_concentrated():
    """Disjoint chemotypes have no parent structure, so the image should not collapse to a few molecules."""
    result = compare(_frame(_many_chains()), _frame(_many_amines()))

    image = result.image_right_in_left
    assert image.image_n > 1
    assert image.median_similarity < 0.3  # and nothing it points at is actually a near neighbour


def test_neighbour_image_threshold_drops_distant_matches():
    """Every molecule has a nearest neighbour; only some have a *close* one."""
    loose = compare(_frame(_many_chains()), _frame(_many_amines()))
    strict = compare(_frame(_many_chains()), _frame(_many_amines()), nn_threshold=0.5)

    assert loose.image_right_in_left.dropped == 0
    assert strict.image_right_in_left.dropped == strict.image_right_in_left.source_n
    assert strict.image_right_in_left.image_n == 0


def test_neighbour_image_is_computed_on_full_sets_not_the_probe_sample():
    """Sampling would shrink the image for reasons unrelated to chemistry."""
    result = compare(_frame(_parents()), _frame(_analogs()), max_samples=5)

    assert result.image_right_in_left.source_n == 24  # all of them, despite max_samples=5
    assert result.image_left_in_right.source_n == 6


def test_report_shows_both_directions_of_the_neighbour_image():
    result = compare(_frame(_CHAINS), _frame(_AMINES), left_name="chains", right_name="amines")
    text = "\n".join(result.report_lines())

    assert "nearest-neighbour image" in text
    assert "amines -> chains" in text
    assert "chains -> amines" in text
    assert "fan-in" in text


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
