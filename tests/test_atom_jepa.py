import sys
import types

import numpy as np
import polars as pl

from representations.atom_jepa import AtomJEPA, AtomJEPABio


class _Stub:
    loaded: list[str] = []

    @classmethod
    def from_pretrained(cls, variant):
        cls.loaded.append(variant)
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self

    def embed(self, smiles):
        return np.ones((len(smiles), 256), dtype=np.float32)


def _install_stub(monkeypatch):
    _Stub.loaded = []
    monkeypatch.setitem(sys.modules, "atom_jepa", types.SimpleNamespace(AtomJEPA=_Stub))


def test_shape_and_invalid_smiles_nan(monkeypatch):
    _install_stub(monkeypatch)
    out = AtomJEPA().transform(pl.Series(["CCO", "not a smiles", "c1ccncc1"]))
    assert out.shape == (3, 256)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[[0, 2]]).all()


def test_variants_load_their_encoder(monkeypatch):
    _install_stub(monkeypatch)
    AtomJEPA().transform(pl.Series(["CCO"]))
    AtomJEPABio().transform(pl.Series(["CCO"]))
    assert _Stub.loaded == ["molecules", "bio"]


def test_registered():
    from data_tools.inputs import INPUT_REGISTRY

    assert "atom_jepa" in INPUT_REGISTRY and "atom_jepa_bio" in INPUT_REGISTRY
