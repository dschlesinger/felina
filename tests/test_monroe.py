"""Monroe checkpoint location, auto-download and embedding mapping. Never downloads."""

import hashlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from representations import monroe as monroe_mod


def _reload(monkeypatch, home):
    if home is None:
        monkeypatch.delenv("MONROE_HOME", raising=False)
    else:
        monkeypatch.setenv("MONROE_HOME", home)
    return importlib.reload(monroe_mod)


@pytest.fixture(autouse=True)
def _restore_module(monkeypatch):
    yield
    monkeypatch.undo()
    importlib.reload(monroe_mod)


def test_default_checkpoint_dir_is_relative_monroe(monkeypatch):
    assert _reload(monkeypatch, None)._CHECKPOINT_DIR == Path(".monroe")


def test_empty_monroe_home_falls_back_to_default(monkeypatch):
    assert _reload(monkeypatch, "")._CHECKPOINT_DIR == Path(".monroe")


def test_monroe_home_overrides_dir(monkeypatch, tmp_path):
    assert _reload(monkeypatch, str(tmp_path))._CHECKPOINT_DIR == tmp_path


def _pin(monkeypatch, content: bytes) -> None:
    """Make `content` the expected weights file, so ensure_checkpoint accepts it without a download."""
    monkeypatch.setattr(monroe_mod, "_WEIGHTS_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(monroe_mod, "_WEIGHTS_BYTES", len(content))


def test_complete_checkpoint_is_not_redownloaded(monkeypatch, tmp_path):
    _pin(monkeypatch, b"weights")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"weights")

    def no_download(*args, **kwargs):
        raise AssertionError("download attempted for a complete checkpoint")

    monkeypatch.setattr(monroe_mod, "_fetch", no_download)
    assert monroe_mod.ensure_checkpoint(tmp_path) == tmp_path


def test_truncated_weights_are_redownloaded(monkeypatch, tmp_path):
    """A partial file from an interrupted earlier run repairs itself rather than failing to load."""
    _pin(monkeypatch, b"weights")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"weig")  # truncated
    calls = []

    def fake_fetch(url, dest, desc):
        calls.append(dest.name)
        dest.write_bytes(b"weights")

    monkeypatch.setattr(monroe_mod, "_fetch", fake_fetch)
    monroe_mod.ensure_checkpoint(tmp_path)
    assert calls == ["weights.pt"]
    assert (tmp_path / "weights.pt").read_bytes() == b"weights"


def test_download_with_a_wrong_digest_is_rejected(monkeypatch, tmp_path):
    _pin(monkeypatch, b"weights")
    monkeypatch.setattr(monroe_mod, "_fetch", lambda url, dest, desc: dest.write_bytes(b"something else"))
    with pytest.raises(RuntimeError, match="sha256"):
        monroe_mod.ensure_checkpoint(tmp_path)
    assert not (tmp_path / "weights.pt").exists()  # a bad file is not left behind


def test_missing_config_is_fetched(monkeypatch, tmp_path):
    _pin(monkeypatch, b"weights")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    fetched = []
    monkeypatch.setattr(monroe_mod, "_fetch", lambda url, dest, desc: (fetched.append(dest.name), dest.write_text("{}")))
    monroe_mod.ensure_checkpoint(tmp_path)
    assert fetched == ["config.json"]


def test_is_flat():
    assert monroe_mod._is_flat(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert not monroe_mod._is_flat(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]]))


def test_parse_spec_and_cache_name(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"weights")
    assert monroe_mod.parse_spec("monroe") is None
    assert monroe_mod.parse_spec(f"monroe[{tmp_path}]") == tmp_path
    stem = monroe_mod.cache_name(f"monroe[{tmp_path}]")
    assert stem.startswith("monroe_") and stem.endswith(f"_{len(b'weights')}_{int((tmp_path / 'weights.pt').stat().st_mtime)}")


def test_registered_and_known_input():
    from data_tools.inputs import INPUT_REGISTRY, is_known_input

    assert "monroe" in INPUT_REGISTRY
    assert is_known_input("monroe") and is_known_input("monroe[/some/checkpoint]")


class _Encoder:
    """Stands in for the GRIT encoder: one constant vector per graph in the batch."""

    def __init__(self):
        import torch

        self._p = torch.zeros(1)

    def parameters(self):
        return iter([self._p])

    def __call__(self, batch):
        import torch

        return torch.arange(len(batch), dtype=torch.float32).unsqueeze(1).repeat(1, 720), None


def _graph(z: float) -> dict:
    return {"pos_rdkit": np.array([[0.0, 0.0, z]], dtype=np.float32)}


def _stub_monroe(monkeypatch, built):
    """Replace the monroe package's featurizer and the PyG batcher; the encoder is injected directly."""
    pytest.importorskip("torch")
    embed = types.SimpleNamespace(
        featurize_smiles=lambda smiles: [(smi, graph) for smi, graph in built if smi in set(smiles)],
        to_pyg=lambda graph: graph,
    )
    monkeypatch.setitem(sys.modules, "monroe", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "monroe.eval", types.SimpleNamespace(embed=embed))
    monkeypatch.setitem(sys.modules, "monroe.eval.embed", embed)
    data = types.SimpleNamespace(Batch=types.SimpleNamespace(from_data_list=lambda graphs: _FakeBatch(graphs)))
    monkeypatch.setitem(sys.modules, "torch_geometric", types.SimpleNamespace(data=data))
    monkeypatch.setitem(sys.modules, "torch_geometric.data", data)


class _FakeBatch(list):
    def to(self, device):
        return self


def test_transform_maps_by_smiles_and_nans_failures(monkeypatch):
    """Molecules that fail featurization or only get a 2D layout must come back NaN, not another molecule's row."""
    built = [("CCO", _graph(1.0)), ("c1ccccc1", _graph(0.0))]  # benzene gets a flat fallback
    _stub_monroe(monkeypatch, built)

    rep = monroe_mod.Monroe()
    rep._encoder = _Encoder()
    out = rep.transform(pl.Series(["CCO", "not_a_smiles", "c1ccccc1", "CCO"]))

    assert out.shape == (4, 720)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()  # unfeaturizable
    assert np.isnan(out[2]).all()  # 2D fallback
    assert np.array_equal(out[0], out[3])  # the same molecule twice gets the same row
