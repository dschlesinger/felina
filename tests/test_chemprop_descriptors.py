"""Extra molecule features for the chemprop backend: row alignment and non-finite handling."""

import numpy as np
import polars as pl

import models.training.chemprop as backend


def _table() -> pl.DataFrame:
    return pl.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CC(=O)O"], "y": [1.0, 2.0, 3.0]})


def test_no_inputs_adds_no_flags(tmp_path):
    assert backend.descriptor_flags(_table(), None, tmp_path) == []
    assert backend.descriptor_flags(_table(), [], tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_writes_one_row_per_table_row_under_arr_0_and_returns_the_flag(tmp_path, monkeypatch):
    features = np.arange(6, dtype=np.float64).reshape(3, 2)
    seen = {}

    def fake_featurize(df, names):
        seen["smiles"], seen["names"] = df["SMILES"].to_list(), names
        return features

    monkeypatch.setattr(backend, "featurize", fake_featurize)
    flags = backend.descriptor_flags(_table(), ["morgan", "rdkit"], tmp_path)

    path = tmp_path / "descriptors.npz"
    assert flags == ["--descriptors-path", str(path)]
    assert seen == {"smiles": ["CCO", "c1ccccc1", "CC(=O)O"], "names": ["morgan", "rdkit"]}
    loaded = np.load(path)["arr_0"]
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, features)


def test_non_finite_values_are_zeroed_and_no_row_is_dropped(tmp_path, monkeypatch, capsys):
    features = np.array([[1.0, np.nan], [np.inf, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(backend, "featurize", lambda df, names: features)
    backend.descriptor_flags(_table(), ["morgan"], tmp_path)

    loaded = np.load(tmp_path / "descriptors.npz")["arr_0"]
    assert loaded.shape == (3, 2)
    np.testing.assert_array_equal(loaded, [[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    assert "zeroed" in capsys.readouterr().err
