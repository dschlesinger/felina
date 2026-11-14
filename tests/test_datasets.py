# tests/test_datasets.py
"""The dataset registry: discovery of data_tools.dataset_defs, and what `data fetch` does with a spec."""

from pathlib import Path

import pytest

from data_tools import data_cli
from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.datasets import registry

EXPECTED = [
    "train",
    "test",
    "test_unblinded",
    "tdi_train",
    "emax_train",
    "single_concentration_train",
    "pubchem_aid1851",
    "p450_family",
    "expansionrx",
    "tdc_hlm",
    "tdc_rlm",
    "tdc_clearance_microsome_az",
    "tdc_half_life_obach",
    "tdc_cyp2c9_substrate",
    "tdc_cyp2d6_substrate",
    "tdc_cyp3a4_substrate",
]


def test_registry_discovers_every_dataset_in_order():
    assert list(registry()) == EXPECTED


def test_a_dataset_package_contributes_one_dataset_not_one_per_submodule():
    # pubchem_aid1851 and p450_family are packages: their fetch/derive code sits beside the spec. Discovery must
    # take the package's SPEC and leave its implementation alone.
    import data_tools.dataset_defs.p450_family as p450

    assert Path(p450.__file__).name == "__init__.py", "p450_family should be a package holding its own code"
    assert [n for n in registry() if n.startswith("p450")] == ["p450_family"]
    assert sorted(registry()["p450_family"].sources) == ["bindingdb", "chembl"]


def test_dataset_specific_code_lives_under_dataset_defs():
    import data_tools

    shared = {p.stem for p in Path(data_tools.__file__).parent.glob("*.py")}
    assert not shared & {"pubchem", "family_datasets", "family_heads"}, "a dataset's code escaped dataset_defs"


def test_every_spec_is_self_consistent():
    for name, spec in registry().items():
        assert spec.name == name
        assert spec.description, f"{name} has no description"
        assert spec.fetched, f"{name} fetches nothing"
        assert len(set(spec.files)) == len(spec.files), f"{name} lists a file twice"
        assert (spec.derived == ()) == (spec.derive is None), f"{name}: derived files and derive step disagree"
        assert spec.primary == spec.files[-1]


def test_source_names_are_unique_across_datasets():
    owners: dict[str, str] = {}
    for name, spec in registry().items():
        for source in spec.sources:
            assert source not in owners, f"{source!r} defined by both {owners[source]} and {name}"
            owners[source] = name


def test_sources_read_a_file_their_dataset_writes():
    for name, spec in registry().items():
        for source_name, source in spec.sources.items():
            assert source.filename in spec.files, f"{name}.{source_name} reads {source.filename}, which it never writes"
            assert source.kind in ("inhouse", "external")
            assert source.task in ("regression", "classification")
            assert source.heads, f"{name}.{source_name} declares no heads"


def test_merge_sources_come_from_the_registry():
    from data_tools.merge_datasets import SOURCES

    assert list(SOURCES) == [s for spec in registry().values() for s in spec.sources]


def test_spec_fills_in_the_source_filename():
    spec = DatasetSpec(
        name="x",
        group="g",
        description="d",
        fetched=("raw.csv",),
        fetch=lambda _: None,
        derived=("ready.csv",),
        derive=lambda _: None,
        sources={"a": Source(kind="external", heads=[(r"^(Y)$", "A")])},
    )
    assert spec.sources["a"].filename == "ready.csv"  # the model-ready file, not the raw one


def test_spec_rejects_derived_files_with_no_derive_step():
    with pytest.raises(ValueError, match="no derive step"):
        DatasetSpec(
            name="x", group="g", description="d",
            fetched=("raw.csv",), fetch=lambda _: None, derived=("ready.csv",),
        )


def _spy_spec(calls: list[str]) -> DatasetSpec:
    def fetch(data_dir: Path) -> None:
        calls.append("fetch")
        (data_dir / "raw.csv").write_text("SMILES\nC\n")

    def derive(data_dir: Path) -> None:
        calls.append("derive")
        (data_dir / "ready.csv").write_text("SMILES\nC\n")

    return DatasetSpec(
        name="spy", group="g", description="d",
        fetched=("raw.csv",), fetch=fetch, derived=("ready.csv",), derive=derive,
    )


def _run(argv: list[str], spec: DatasetSpec, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data_cli, "registry", lambda: {spec.name: spec})
    data_cli.main(["--data-dir", str(tmp_path), *argv])


def test_fetch_skips_the_download_when_the_raw_files_are_there(monkeypatch, tmp_path):
    calls: list[str] = []
    spec = _spy_spec(calls)
    (tmp_path / "raw.csv").write_text("SMILES\nC\n")

    _run(["fetch", "spy"], spec, monkeypatch, tmp_path)

    # Derive runs every time, so a change to the reshape is picked up without a refetch.
    assert calls == ["derive"]


def test_fetch_downloads_when_a_raw_file_is_missing(monkeypatch, tmp_path):
    calls: list[str] = []
    _run(["fetch", "spy"], _spy_spec(calls), monkeypatch, tmp_path)
    assert calls == ["fetch", "derive"]


def test_redownload_fetches_even_when_the_raw_files_are_there(monkeypatch, tmp_path):
    calls: list[str] = []
    spec = _spy_spec(calls)
    (tmp_path / "raw.csv").write_text("SMILES\nC\n")

    _run(["fetch", "spy", "--redownload"], spec, monkeypatch, tmp_path)

    assert calls == ["fetch", "derive"]
