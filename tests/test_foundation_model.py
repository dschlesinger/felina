"""The `foundation-model` / `fm` CLI: finding fine-tunes, choosing what to embed, and fitting tables to the terminal."""

import argparse
from pathlib import Path

import pytest

from models import foundation_model as fm


def _touch(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A results tree with fine-tunes in run folders and in the per-kind homes; aliases saved under tmp_path."""
    monkeypatch.chdir(tmp_path)  # foundation-path-alias.csv is relative to the working directory
    runs = tmp_path / "runs"
    for kind in fm.KINDS.values():
        monkeypatch.setenv(kind.home_env, str(tmp_path / kind.name))
    _touch(runs / "0007__train__model=unimol_datasets=challenge" / "lr_0.0002_seed_0" / "model_0.pth")
    _touch(runs / "0007__train__model=unimol_datasets=challenge" / "lr_0.0002_seed_1" / "model_0.pth")
    _touch(runs / "0012__train__model=chemeleon_datasets=challenge" / "model_0" / "best.pt", 20)
    (runs / "0013__eval__models=xgb").mkdir()
    (runs / "0014__train__model=atom_jepa").mkdir()  # still training: no weights yet
    _touch(tmp_path / "chemprop" / "challenge" / "seed0" / "model_0" / "best.pt")
    (tmp_path / "chemprop" / "challenge" / "seed1").mkdir()  # never finished
    _touch(tmp_path / "chemprop" / "chemeleon" / "challenge" / "seed0" / "model_0" / "best.pt")
    _touch(tmp_path / "atom_jepa_bio" / "challenge" / "lr_0.001_seed_0" / "encoder.pt")
    return runs


def test_catalog_trainables_match_the_training_backends():
    from models.training import BACKENDS

    assert set(fm.TRAINABLE) == set(BACKENDS)
    for f in fm.CATALOG.values():
        assert f.kind is None or f.kind in fm.KINDS


def test_discover_finds_run_folders_and_homes(machine):
    found = fm.discover(machine)
    by_alias = {f.alias: f for f in found}
    assert set(by_alias) == {
        "unimol_0007_lr0.0002_s0",
        "unimol_0007_lr0.0002_s1",
        "chemeleon_0012",
        "challenge_0",
        "chemeleon_0",  # results/chemprop/chemeleon/challenge/seed0
        "challenge_lr0.001_s0",
    }
    assert by_alias["chemeleon_0012"].model == "chemeleon"
    assert by_alias["chemeleon_0012"].spec.startswith("chemprop[")
    assert by_alias["unimol_0007_lr0.0002_s0"].run == "lr_0.0002_seed_0"
    assert by_alias["chemeleon_0012"].run == ""  # one weights file in the folder
    models = {f.alias: f.model for f in found}
    assert models["challenge_0"] == "chemprop" and models["chemeleon_0"] == "chemeleon"
    assert models["challenge_lr0.001_s0"] == "atom_jepa_bio"


def test_aliases_are_stable_across_listings(machine):
    first = {f.weights: f.alias for f in fm.discover(machine)}
    _touch(machine / "0020__train__model=unimol" / "model_0.pth")
    second = {f.weights: f.alias for f in fm.discover(machine)}
    assert all(second[w] == a for w, a in first.items())


def _args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"model": None, "only": None, **kw})


def test_embed_targets_by_model_alias_and_only(machine, monkeypatch):
    monkeypatch.setattr(fm, "discover", lambda runs_dir=None: fm._run_finetunes(machine) + fm._home_finetunes())
    unimol = fm._embed_targets(_args(model=["unimol"]))
    assert unimol[0] == ("unimol pretrained", "unimol")
    assert [label for label, _ in unimol[1:]] == ["unimol unimol_0007_lr0.0002_s0", "unimol unimol_0007_lr0.0002_s1"]
    assert fm._embed_targets(_args(model=["unimol"], only="pretrained")) == [("unimol pretrained", "unimol")]
    assert len(fm._embed_targets(_args(model=["unimol"], only="finetuned"))) == 2
    # chemprop has no pretrained encoder: its fine-tunes only
    assert all(spec.startswith("chemprop[") for _, spec in fm._embed_targets(_args(model=["chemprop"])))
    (label, spec), = fm._embed_targets(_args(model=["chemeleon_0012"]))
    assert label == "chemeleon chemeleon_0012" and spec.startswith("chemprop[")
    everything = fm._embed_targets(_args())
    assert {s for _, s in everything} >= set(fm.PRETRAINED) and len(everything) == len(fm.PRETRAINED) + 6
    with pytest.raises(SystemExit):
        fm._embed_targets(_args(model=["nope"]))


def test_coverage_counts_cached_molecules(tmp_path):
    (tmp_path / "chemeleon.npy").write_bytes(b"")
    (tmp_path / "chemeleon.idx").write_text("CCO\nCCN")
    import polars as pl

    coverage = fm.Coverage(tmp_path, lambda: pl.DataFrame({"SMILES": ["CCO", "CCN", "CCC"]}), "x")
    assert coverage.cell("chemeleon") == "[yellow]2/3[/]"
    assert coverage.cell("unimolv2_310m") == "[dim]none[/]"
    missing = fm.Coverage(tmp_path, lambda: (_ for _ in ()).throw(FileNotFoundError("no data")), "x")
    assert missing.cell("chemeleon") == "2 mols"


def test_render_drops_whole_columns_in_order(monkeypatch):
    monkeypatch.setattr(fm.console, "width", 40)
    columns = [fm.Column("Alias"), fm.Column("Size", "right"), fm.Column("Run"), fm.Column("Path", fold=1 << 30)]
    rows = [["unimol_0007_lr0.0002_s0", "2.9 MB", "lr_0.0002_seed_0", "results/runs/" + "x" * 60]]
    headers = [c.header for c in fm._render(columns, rows, drop=["Path", "Run"]).columns]
    assert headers == ["Alias", "Size"]
    kept = [c.header for c in fm._render(columns, rows, drop=["Path", "Run"], keep={"Path"}).columns]
    assert kept == ["Alias", "Size", "Path"]
    monkeypatch.setattr(fm.console, "width", 200)
    assert [c.header for c in fm._render(columns, rows, drop=["Path", "Run"]).columns] == ["Alias", "Size", "Run", "Path"]
