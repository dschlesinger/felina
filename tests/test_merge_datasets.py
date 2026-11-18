# tests/test_merge_datasets.py
"""Run-time merge of selected datasets into per-source heads."""

import polars as pl
import pytest

from data_tools.dataset_spec import Source
from data_tools.merge_datasets import SOURCES, merge, parse_dataset_specs

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
IBUPROFEN = "CC(C)CC1=CC=C(C=C1)C(=O)O"
PHENOL = "C1=CC=C(C=C1)O"
TOLUENE = "CC1=CC=CC=C1"

# The shipped challenge has no external, classification or log-scaled source, but the merge still
# supports all three. They are exercised through synthetic sources registered for the test, so the code
# path stays covered whatever datasets the challenge of the day ships.
_EXTRA_SOURCES = {
    "ext": Source(
        kind="external",
        heads=[(r"^(pEC50)_ext$", "{}-Ext"), (r"^(Emax_log2FC)_ext$", "{}-Ext")],
        filename="external.csv",
    ),
    "ext_sparse": Source(kind="external", heads=[(r"^(pEC50)_sparse$", "{}-Sparse")], filename="external.csv"),
    "binary": Source(
        kind="external", heads=[(r"^(induces)$", "{}-Binary")], task="classification", filename="binary.csv"
    ),
    "clint": Source(kind="external", heads=[(r"^(CLint)$", "{}-Clint")], log10p1=True, filename="clint.csv"),
}


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("data_tools.merge_datasets.SOURCES", {**SOURCES, **_EXTRA_SOURCES})
    names = ["OCNT-1", "OCNT-2", "OCNT-3"]
    smiles = [ASPIRIN, CAFFEINE, IBUPROFEN]
    pl.DataFrame(
        {"Molecule_Name": names, "SMILES": smiles, "pEC50": [5.0, None, 6.0]}
    ).write_csv(tmp_path / "train.csv")
    pl.DataFrame(
        {
            "Molecule_Name": names + ["OCNT-4"],
            "SMILES": smiles + [PHENOL],
            "Emax_log2FC": [1.5, None, 2.0, 2.5],
            "Emax_vs_pos_ctrl": [0.5, 0.7, None, 0.9],
        }
    ).write_csv(tmp_path / "emax_train.csv")
    pl.DataFrame(
        {
            "SMILES": [ASPIRIN, TOLUENE, "CCO"],
            "pEC50_ext": [7.0, 5.0, None],
            "pEC50_sparse": [None, None, 3.0],
            "Emax_log2FC_ext": [6.0, None, None],
        }
    ).write_csv(tmp_path / "external.csv")
    return tmp_path


def _merge(data_dir, specs):
    return merge(parse_dataset_specs(specs), data_dir=data_dir, test_path=None, split_strategy="random")


def test_challenge_only_has_scored_heads(data_dir):
    table, heads, scored, _ = _merge(data_dir, [])
    assert heads == scored == ["pEC50-PXR"]
    assert len(table) == 2  # caffeine carries no scored label, so the challenge source drops it


def test_heads_are_split_by_source(data_dir):
    table, heads, _, _ = _merge(data_dir, ["emax", "ext"])
    assert {"Emax_log2FC-PXR", "Emax_vs_pos_ctrl-PXR", "pEC50-Ext", "Emax_log2FC-Ext"} <= set(heads)
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert aspirin["pEC50-PXR"] == 5.0
    assert aspirin["Emax_log2FC-PXR"] == 1.5
    assert aspirin["pEC50-Ext"] == 7.0  # same molecule, different assay: its own head, never pooled


def test_target_filter_restricts_heads(data_dir):
    _, heads, _, _ = _merge(data_dir, ["emax:Emax_log2FC"])
    assert "Emax_log2FC-PXR" in heads and "Emax_vs_pos_ctrl-PXR" not in heads


def test_external_only_molecules_are_train_and_keep_smiles(data_dir):
    table, _, _, _ = _merge(data_dir, ["ext"])
    toluene = table.filter(pl.col("SMILES") == TOLUENE).row(0, named=True)
    assert toluene["split"] == "train"
    assert toluene["pEC50-PXR"] is None


def test_source_without_labels_adds_no_rows(data_dir):
    table, _, _, _ = _merge(data_dir, ["ext_sparse"])
    assert "CCO" in table["SMILES"].to_list()  # the only molecule labelled for this head
    assert TOLUENE not in table["SMILES"].to_list()  # a row labelled only for the other head must not leak in


def test_inhouse_only_molecule_is_train(data_dir):
    table, _, _, _ = _merge(data_dir, ["emax"])
    phenol = table.filter(pl.col("SMILES") == PHENOL).row(0, named=True)
    assert phenol["split"] == "train"


def test_split_ignores_dataset_choice(data_dir):
    held = lambda specs: sorted(  # noqa: E731
        _merge(data_dir, specs)[0].filter(pl.col("split") != "train")["SMILES"].to_list()
    )
    assert held([]) == held(["emax", "ext", "ext_sparse"])


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        parse_dataset_specs(["nope"])


def _scaled_files(data_dir):
    pl.DataFrame({"SMILES": [ASPIRIN, ASPIRIN, TOLUENE, "CCO"], "induces": [1, 0, 0, 1]}).write_csv(
        data_dir / "binary.csv"
    )
    pl.DataFrame({"SMILES": [ASPIRIN, IBUPROFEN], "CLint": [9.0, 0.0]}).write_csv(data_dir / "clint.csv")


def test_binary_source_is_classification_and_settles_ties(data_dir):
    _scaled_files(data_dir)
    table, _, _, tasks = _merge(data_dir, ["binary"])
    assert tasks["induces-Binary"] == "classification" and tasks["pEC50-PXR"] == "regression"
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert aspirin["induces-Binary"] == 1.0  # duplicate labels 1 and 0 average to 0.5, ties go positive


def test_clint_is_log_scaled(data_dir):
    _scaled_files(data_dir)
    table, heads, _, tasks = _merge(data_dir, ["clint"])
    assert "CLint-Clint" in heads
    assert tasks["CLint-Clint"] == "regression"
    ibuprofen = table.filter(pl.col("SMILES") == IBUPROFEN).row(0, named=True)
    assert ibuprofen["CLint-Clint"] == 0.0  # log10(0 + 1)
    aspirin = table.filter(pl.col("SMILES") == ASPIRIN).row(0, named=True)
    assert abs(aspirin["CLint-Clint"] - 1.0) < 1e-9  # log10(9 + 1)


def test_specs_can_omit_challenge(data_dir):
    assert parse_dataset_specs(["emax"], with_challenge=False) == {"emax": None}
    assert list(parse_dataset_specs(["emax"])) == ["challenge", "emax"]


def test_blind_matched_strategy_uses_the_blind_set_as_its_reference(data_dir):
    """The held-out slices imitate the blind set instead of holding out whole scaffolds."""
    test_path = data_dir / "blind.csv"
    pl.DataFrame({"Molecule_Name": ["OCNT-9", "OCNT-10"], "SMILES": [TOLUENE, PHENOL]}).write_csv(
        test_path
    )
    table, _, _, _ = merge(
        parse_dataset_specs([]),
        data_dir=data_dir,
        test_path=test_path,
        split_strategy="blind_matched",
        val_fraction=0.34,
        test_fraction=0.34,
    )
    assert "split" in table.columns
    assert set(table["split"].to_list()) <= {"train", "val", "test"}


def test_blind_matched_strategy_requires_a_reference(data_dir):
    with pytest.raises(ValueError, match="needs the blind set as its reference"):
        merge(
            parse_dataset_specs([]),
            data_dir=data_dir,
            test_path=None,
            split_strategy="blind_matched",
        )
