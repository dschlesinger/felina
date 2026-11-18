"""The standardization modes applied to SMILES before featurization."""

import numpy as np
import polars as pl
import pytest

from data_tools import standardization
from data_tools.inputs import featurize


@pytest.fixture(autouse=True)
def reset_mode(monkeypatch):
    """Each test starts from the default mode and leaves no global behind."""
    monkeypatch.delenv(standardization.ENV_VAR, raising=False)
    before = standardization.get_mode()
    standardization.set_mode(standardization.DEFAULT_MODE)
    yield
    standardization.set_mode(before)


def test_default_mode_is_normal():
    """The default is the behaviour every recorded result was produced under."""
    assert standardization.DEFAULT_MODE == "normal"
    assert standardization.get_mode() == "normal"


def test_modes_are_the_four_documented_ones():
    assert standardization.MODES == ("none", "normal", "advanced", "tautomer")


def test_none_passes_smiles_through_untouched():
    """Not even canonicalization: 'none' means the string reaches the featurizer as written."""
    written = "C(C)O.[Na+]"
    assert standardization.standardize_one(written, "none") == written


def test_normal_canonicalizes():
    assert standardization.standardize_one("C(C)O", "normal") == "CCO"


def test_normal_keeps_the_longest_string_component():
    """Documented wart: 'normal' strips salts by string length, so a bracketed ion can win.

    This is what the project has always done; `advanced` is the mode that picks
    the parent by molecular weight instead.
    """
    assert standardization.standardize_one("CCO.[Na+]", "normal") == "[Na+]"


def test_advanced_takes_the_parent_by_weight():
    assert standardization.standardize_one("CCO.[Na+]", "advanced") == "CCO"


def test_advanced_neutralizes_charges():
    assert standardization.standardize_one("CC(=O)[O-].[Na+]", "advanced") == "CC(=O)O"
    assert standardization.standardize_one("CC(=O)[O-]", "normal") == "CC(=O)[O-]"


@pytest.mark.parametrize(
    "pair",
    [
        ("Oc1ccccn1", "O=c1cccc[nH]1"),  # 2-hydroxypyridine / 2-pyridone
        ("CC(=O)CC(C)=O", "CC(O)=CC(C)=O"),  # keto / enol
        ("CC(=O)N", "CC(O)=N"),  # amide / imidic acid
    ],
)
def test_tautomer_collapses_a_tautomer_pair(pair):
    """Both spellings land on one representative, which `advanced` leaves apart."""
    left, right = (standardization.standardize_one(s, "tautomer") for s in pair)
    assert left == right
    assert standardization.standardize_one(pair[0], "advanced") != standardization.standardize_one(
        pair[1], "advanced"
    )


def test_tautomer_keeps_sp3_stereo():
    """RDKit's enumerator strips stereo by default; methotrexate must keep its alpha carbon.

    Stereoisomers carry different CYP labels here, so the tautomer mode is not
    allowed to flatten them on its way past.
    """
    methotrexate = "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1"
    assert "@" in standardization.standardize_one(methotrexate, "tautomer")


def test_tautomer_keeps_double_bond_stereo():
    """Diethylstilbestrol keeps its double-bond geometry."""
    des = "CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1"
    out = standardization.standardize_one(des, "tautomer")
    assert "/" in out or "\\" in out


def test_tautomer_still_does_everything_advanced_does():
    """It is `advanced` plus a step, so the salt and the charge are handled too."""
    assert standardization.standardize_one("CC(=O)[O-].[Na+]", "tautomer") == "CC(=O)O"


def test_unparsable_smiles_is_returned_unchanged():
    """A molecule RDKit rejects reaches the featurizer as it would under 'none', rather than vanishing."""
    for mode in ("normal", "advanced", "tautomer"):
        assert standardization.standardize_one("not-a-molecule", mode) == "not-a-molecule"


def test_blank_smiles_survives():
    assert standardization.standardize_one("", "advanced") == ""


def test_standardize_many_matches_standardize_one():
    smiles = ["C(C)O", "CC(=O)[O-].[Na+]", "c1ccccc1"]
    assert standardization.standardize_many(smiles, "advanced") == [
        standardization.standardize_one(s, "advanced") for s in smiles
    ]


def test_standardize_column_rewrites_only_that_column():
    df = pl.DataFrame({"SMILES": ["C(C)O"], "label": [1.0]})
    out = standardization.standardize_column(df, "SMILES", "normal")
    assert out["SMILES"].to_list() == ["CCO"]
    assert out["label"].to_list() == [1.0]


def test_standardize_column_without_the_column_is_a_no_op():
    df = pl.DataFrame({"other": ["C(C)O"]})
    assert standardization.standardize_column(df, "SMILES", "advanced").equals(df)


def test_changed_counts_rewrites():
    assert standardization.changed(["a", "b"], ["a", "c"]) == 1


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown standardization mode"):
        standardization.resolve("aggressive")
    with pytest.raises(ValueError, match="Unknown standardization mode"):
        standardization.set_mode("aggressive")


def test_set_mode_is_mirrored_to_the_environment(monkeypatch):
    """Subprocess backends read the mode from the environment, so it has to be written there."""
    import os

    standardization.set_mode("advanced")
    assert os.environ[standardization.ENV_VAR] == "advanced"
    assert standardization.resolve(None) == "advanced"


def test_explicit_mode_overrides_the_process_mode():
    standardization.set_mode("advanced")
    assert standardization.standardize_one("CCO.[Na+]", "normal") == "[Na+]"


def test_featurize_keys_the_cache_on_the_standardized_smiles(tmp_path):
    """Two spellings of one molecule give one cache row and identical features."""
    standardization.set_mode("normal")
    df = pl.DataFrame({"SMILES": ["C(C)O", "CCO"]})
    features = featurize(df, "morgan", tmp_path)

    assert np.array_equal(features[0], features[1])
    assert (tmp_path / "morgan.idx").read_text().splitlines() == ["CCO"]


def test_featurize_under_none_keeps_the_spellings_apart(tmp_path):
    """'none' is the old behaviour: one cache row per spelling."""
    standardization.set_mode("none")
    df = pl.DataFrame({"SMILES": ["C(C)O", "CCO"]})
    featurize(df, "morgan", tmp_path)

    assert sorted((tmp_path / "morgan.idx").read_text().splitlines()) == ["C(C)O", "CCO"]


def test_featurize_argument_overrides_the_process_mode(tmp_path):
    standardization.set_mode("none")
    df = pl.DataFrame({"SMILES": ["C(C)O"]})
    featurize(df, "morgan", tmp_path, standardization="normal")

    assert (tmp_path / "morgan.idx").read_text().splitlines() == ["CCO"]


def test_featurize_rows_follow_the_requested_order(tmp_path):
    """Deduplicating in the cache must not reorder or drop the rows the caller asked for."""
    standardization.set_mode("normal")
    df = pl.DataFrame({"SMILES": ["c1ccccc1", "C(C)O", "c1ccccc1"]})
    features = featurize(df, "morgan", tmp_path)

    assert features.shape[0] == 3
    assert np.array_equal(features[0], features[2])
    assert not np.array_equal(features[0], features[1])
