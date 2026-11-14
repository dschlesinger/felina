# tests/test_dimred.py
"""Dimensionality reduction: TorchPCA matches sklearn, specs parse, fits are cached."""

import numpy as np
import pytest

pytest.importorskip("torch")

from sklearn.decomposition import PCA

from models.dimred import DimRed, TorchPCA, fit, parse_spec


def test_matches_sklearn_pca_up_to_sign():
    X = np.random.default_rng(0).normal(size=(200, 40))
    ours = TorchPCA(n_components=5, device="cpu").fit(X).transform(X)
    ref = PCA(n_components=5).fit_transform(X)
    np.testing.assert_allclose(np.abs(ours), np.abs(ref), atol=1e-3)


def test_output_shape_and_set_params():
    X = np.random.default_rng(1).normal(size=(50, 12))
    pca = TorchPCA(device="cpu")
    pca.set_params(n_components=3)
    assert pca.fit(X).transform(X).shape == (50, 3)


def test_parse_spec():
    assert parse_spec("pca[200]") == DimRed("pca", 200)
    assert parse_spec("pls[8]").supervised
    assert not parse_spec("umap[16]").supervised
    for bad in ["pca", "pca[0]", "tsne[2]", "pca[2.5]"]:
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_pca_clamps_and_caches(tmp_path):
    X = np.random.default_rng(2).normal(size=(30, 8))
    reducer = fit(parse_spec("pca[50]"), X, cache_dir=tmp_path)
    assert reducer.transform(X).shape == (30, 8)
    assert len(list(tmp_path.glob("pca_*.pkl"))) == 1
    again = fit(parse_spec("pca[50]"), X, cache_dir=tmp_path)
    np.testing.assert_allclose(again.transform(X), reducer.transform(X))


def test_pls_needs_labels_and_keys_on_them(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 10))
    y = X[:, 0] + 0.1 * rng.normal(size=40)
    with pytest.raises(ValueError):
        fit(parse_spec("pls[2]"), X, cache_dir=tmp_path)
    assert fit(parse_spec("pls[2]"), X, y, cache_dir=tmp_path).transform(X).shape == (40, 2)
    fit(parse_spec("pls[2]"), X, -y, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("pls_*.pkl"))) == 2
