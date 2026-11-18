# tests/test_datasets.py
"""The dataset registry: discovery of data_tools.dataset_defs, and what `data fetch` does with a spec."""

from pathlib import Path

import pytest

from data_tools import data_cli
from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.datasets import blind_spec, challenge, parse_selector, registry

EXPECTED = [
    "train",
    "emax_train",
    "test",
    "test_unblinded",
    "structure_test",
]


def test_registry_discovers_every_dataset_in_order():
    assert list(registry()) == EXPECTED


def test_a_group_contributes_its_datasets_and_skips_its_private_helpers():
    # pxr/ is a group: the challenge's datasets sit in it beside _reshape.py, the column mapping and derive
    # steps they share. Discovery must recurse into the group, collect the specs, and skip the underscore
    # module rather than demanding a SPEC from it.
    import data_tools.dataset_defs.pxr as pxr

    assert Path(pxr.__file__).name == "__init__.py"
    assert not hasattr(pxr, "SPEC"), "a group holds datasets; it is not one"
    assert list(registry()) == EXPECTED
    assert sorted(registry()["train"].sources) == ["challenge"]
    assert sorted(registry()["emax_train"].sources) == ["emax"]


def test_dataset_specific_code_lives_under_dataset_defs():
    import data_tools

    shared = {p.stem for p in Path(data_tools.__file__).parent.glob("*.py")}
    assert not shared & {"pxr", "reshape"}, "a dataset's code escaped dataset_defs"


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


# --- the scored challenge: the metadata every evaluator reads instead of hardcoding CYP columns -------------


def _spec(**kwargs) -> DatasetSpec:
    base = dict(name="x", group="g", description="d", fetched=("x.csv",), fetch=lambda _: None)
    return DatasetSpec(**(base | kwargs))


def test_challenge_is_the_dataset_declaring_scored():
    spec = challenge()
    assert spec.scored and spec.name == "train"
    assert spec.targets == ("pEC50",)
    # No suffix to trim: the scored column is already the name it reports under.
    assert spec.label_suffix == ""
    assert spec.shorthand("pEC50") == "pEC50"


def test_challenge_dataset_env_var_picks_the_scored_dataset(monkeypatch):
    monkeypatch.setenv("CHALLENGE_DATASET", "train")
    assert challenge().name == "train"


def test_challenge_dataset_env_var_rejects_an_unscored_dataset(monkeypatch):
    monkeypatch.setenv("CHALLENGE_DATASET", "test")
    with pytest.raises(ValueError, match="does not declare scored=True"):
        challenge()


def test_challenge_dataset_env_var_rejects_an_unknown_name(monkeypatch):
    monkeypatch.setenv("CHALLENGE_DATASET", "nope")
    with pytest.raises(ValueError, match="not a known dataset"):
        challenge()


def test_blind_spec_is_the_dataset_the_challenge_names():
    assert blind_spec().name == "test"


def test_a_scored_dataset_must_name_its_targets():
    with pytest.raises(ValueError, match="must name its targets"):
        _spec(scored=True)


def test_targets_may_not_repeat_a_column():
    with pytest.raises(ValueError, match="repeats a column"):
        _spec(targets=("a", "a"))


def test_shorthand_trims_the_label_suffix():
    spec = _spec(targets=("CYP2D6_pIC50",), label_suffix="_pIC50")
    assert spec.shorthand("CYP2D6_pIC50") == "CYP2D6"
    assert spec.shorthand("other") == "other"  # nothing to trim


def test_resolve_targets_defaults_to_every_target():
    spec = _spec(targets=("a_p", "b_p"), label_suffix="_p")
    assert spec.resolve_targets(None) == ["a_p", "b_p"]
    assert spec.resolve_targets([]) == ["a_p", "b_p"]


def test_resolve_targets_accepts_shorthand_and_full_names_and_keeps_spec_order():
    spec = _spec(targets=("a_p", "b_p", "c_p"), label_suffix="_p")
    assert spec.resolve_targets(["c", "a_p"]) == ["a_p", "c_p"]
    assert spec.resolve_targets(["b", "b_p"]) == ["b_p"]  # deduplicated


def test_resolve_targets_names_the_unknown_selector_and_the_choices():
    spec = _spec(targets=("a_p",), label_suffix="_p")
    with pytest.raises(ValueError, match="has no target\\(s\\) zzz; choose from a"):
        spec.resolve_targets(["zzz"])


def test_parse_selector_splits_a_name_from_its_targets():
    spec, targets = parse_selector("train:pEC50")
    assert spec.name == "train"
    assert targets == ["pEC50"]


def test_parse_selector_without_targets_takes_them_all():
    assert parse_selector("train")[1] == list(challenge().targets)


def test_parse_selector_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset 'nope'"):
        parse_selector("nope")


def test_parse_selector_rejects_a_dataset_that_is_not_scored():
    with pytest.raises(ValueError, match="not a scored dataset"):
        parse_selector("test")


def test_resolve_targets_accepts_an_unambiguous_tail_of_a_shorthand():
    # The documented tail shorthand, on this challenge's one target: `EC50` can only mean pEC50.
    assert challenge().resolve_targets(["EC50"]) == ["pEC50"]


def test_resolve_targets_ignores_case():
    assert challenge().resolve_targets(["pec50"]) == ["pEC50"]


def test_resolve_targets_rejects_a_tail_that_names_several_targets():
    spec = _spec(targets=("a_x_p", "b_x_p"), label_suffix="_p")
    with pytest.raises(ValueError, match="has no target"):
        spec.resolve_targets(["x_p"])  # tail of both shorthands, so it names neither
