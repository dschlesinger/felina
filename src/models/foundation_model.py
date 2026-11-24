"""The `foundation-model` (or `fm`) CLI: which foundation models exist, which fine-tunes are on this machine,
embedding molecules with them, and training new fine-tunes.

  fm avail                                  # every foundation model: pretrained weights, fine-tunes, sizes
  fm finetunes [--models unimol chemeleon]  # fine-tuned weights on disk: alias, size, embedding status, path
  fm embed [--model unimol] [--dataset NAME ...] [--only pretrained|finetuned]
  fm train --model chemeleon --datasets challenge pubchem:CYP1A2   # see `fm train --model X --help`

Fine-tunes are found in two places: run folders under <RESULTS_FOLDER>/runs written by `fm train`
(`<id>__train__model=<model>_...`), and the older per-kind homes (results/chemprop, results/unimol, ...;
override each with its $<KIND>_MODEL_HOME, $FOUNDATION_MODEL_HOME for chemprop).
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import polars as pl
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from data_tools import standardization
from data_tools.paths import DATA_DIR, FEATURE_DIR, RESULTS_DIR
from models.run_log import RUNS_DIR, clean_interrupt
from representations.atom_jepa import cache_name as atom_jepa_cache_name
from representations.atom_jepa import cache_prefix as atom_jepa_cache_prefix
from representations.chemprop_weights import cache_name as chemprop_cache_name
from representations.chemprop_weights import cache_prefix as chemprop_cache_prefix
from representations.foundation_alias import ensure_aliases
from representations.unimol import MODEL_SIZE as UNIMOL_SIZE
from representations.unimol import PRETRAINED_CACHE_STEM as UNIMOL_PRETRAINED_CACHE_STEM
from representations.unimol import cache_name as unimol_cache_name
from representations.unimol import cache_prefix as unimol_cache_prefix

_MODEL_CHARS = 10

console = Console()


# --------------------------------------------------------------------------------------------------------------
# Kinds: how a family of fine-tuned weights is laid out on disk and named as an input spec
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """A family of trained models: where they live and how their runs and weights are laid out.

    A run directory (a chemprop seed, a Uni-Mol learning-rate and seed pair) holds one weights file; a model is the
    directory holding its runs.
    """

    name: str
    default_home: Path
    home_env: str
    run_pattern: re.Pattern  # matches run directory names; its groups are all numeric and sort the runs
    alias_suffix: str  # formatted with the run_pattern groups, appended to the model name to form an alias
    weights_file: Path  # relative to a run directory
    weights_glob: str  # finds the weights anywhere inside a `fm train` run folder
    cache_name: Callable[[str], str]
    cache_prefix: Callable[[str], str]

    def home(self) -> Path:
        """Model directory: the environment variable if set, else the default."""
        return Path(os.environ.get(self.home_env) or self.default_home)

    def spec(self, weights: Path) -> str:
        return f"{self.name}[{weights}]"

    def run_dir(self, weights: Path) -> Path:
        return weights.parents[len(self.weights_file.parts) - 1]

    def run_key(self, run: Path) -> tuple[float, ...]:
        return tuple(float(g) for g in self.run_pattern.fullmatch(run.name).groups())

    def alias_for(self, model: str, weights: Path) -> str:
        return f"{model}_{self.alias_suffix.format(**self.run_pattern.fullmatch(self.run_dir(weights).name).groupdict())}"


_LR_SEED = re.compile(r"lr_(?P<lr>[0-9.]+(?:e[+-]?\d+)?)_seed_(?P<seed>\d+)")

KINDS = {
    "chemprop": Kind(
        "chemprop",
        RESULTS_DIR / "chemprop",
        "FOUNDATION_MODEL_HOME",
        re.compile(r"seed(?P<seed>\d+)"),
        "{seed}",
        Path("model_0/best.pt"),
        "best*.pt",
        chemprop_cache_name,
        chemprop_cache_prefix,
    ),
    "unimol": Kind(
        "unimol",
        RESULTS_DIR / "unimol",
        "UNIMOL_MODEL_HOME",
        _LR_SEED,
        "lr{lr}_s{seed}",
        Path("model_0.pth"),
        "model_0.pth",
        unimol_cache_name,
        unimol_cache_prefix,
    ),
    **{
        name: Kind(
            name,
            RESULTS_DIR / name,
            f"{name.upper()}_MODEL_HOME",
            _LR_SEED,
            "lr{lr}_s{seed}",
            Path("encoder.pt"),
            "encoder.pt",
            atom_jepa_cache_name,
            atom_jepa_cache_prefix,
        )
        for name in ("atom_jepa", "atom_jepa_bio")
    },
}
HOME_ENV = KINDS["chemprop"].home_env
DEFAULT_HOME = KINDS["chemprop"].default_home


def list_models(home: Path, kind: Kind = KINDS["chemprop"]) -> list[tuple[str, list[Path]]]:
    """Return (name, weights path of each run) for each model under home.

    A model is any directory (at any depth) holding run subdirectories (seed<N> for chemprop,
    lr_<rate>_seed_<N> for Uni-Mol); its name is its path relative to home.
    Runs are in numeric order, and the path is the run's weights file whether or not it exists yet.
    """
    by_model: dict[Path, list[Path]] = {}
    for path in home.rglob("*"):
        if path.is_dir() and kind.run_pattern.fullmatch(path.name):
            by_model.setdefault(path.parent, []).append(path)
    return [
        (
            str(model_dir.relative_to(home)),
            [run / kind.weights_file for run in sorted(runs, key=kind.run_key)],
        )
        for model_dir, runs in sorted(by_model.items())
    ]


# --------------------------------------------------------------------------------------------------------------
# The catalog: every foundation model the project knows
# --------------------------------------------------------------------------------------------------------------


def _hf_hub() -> Path:
    """The Hugging Face hub cache, where atom_jepa downloads its released encoders."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return Path(home) / "hub"


def _unimol_weights() -> list[Path]:
    """The Uni-Mol v2 checkpoint unimol_tools downloads into its weights directory (or $UNIMOL_WEIGHT_DIR)."""
    if os.environ.get("UNIMOL_WEIGHT_DIR"):
        root = Path(os.environ["UNIMOL_WEIGHT_DIR"])
    else:
        spec = find_spec("unimol_tools")  # located without importing it: importing loads torch
        if spec is None or not spec.submodule_search_locations:
            return []
        root = Path(next(iter(spec.submodule_search_locations))) / "weights"
    return [root / "modelzoo" / UNIMOL_SIZE.upper() / "checkpoint.pt"]


def _atom_jepa_weights(variant: str) -> Callable[[], list[Path]]:
    def find() -> list[Path]:
        snapshots = _hf_hub() / "models--atom-jepa--atom-jepa" / "snapshots"
        return sorted(snapshots.glob(f"*/{variant}/*")) if snapshots.is_dir() else []

    return find


def _chemeleon_weights() -> list[Path]:
    from representations.chemeleon import _CHECKPOINT_PATH

    return [_CHECKPOINT_PATH]


def _monroe_weights() -> list[Path]:
    from representations.monroe import _CHECKPOINT_DIR

    return [_CHECKPOINT_DIR / "weights.pt", _CHECKPOINT_DIR / "config.json"]


@dataclass(frozen=True)
class Foundation:
    """One foundation model: what it is, its pretrained encoder (if any), and whether `fm train` fine-tunes it."""

    name: str
    about: str
    kind: str | None  # the Kind its fine-tunes are stored and embedded as; None when it cannot be fine-tuned
    pretrained: str | None  # the registered input that embeds with the pretrained encoder; None if there is none
    pretrained_weights: Callable[[], list[Path]] | None = None  # where the pretrained weights land once downloaded
    trainable: bool = True

    @property
    def pretrained_stem(self) -> str | None:
        if self.pretrained is None:
            return None
        return UNIMOL_PRETRAINED_CACHE_STEM if self.pretrained == "unimol" else self.pretrained


CATALOG: dict[str, Foundation] = {
    f.name: f
    for f in (
        Foundation("chemprop", "Chemprop D-MPNN, trained from scratch (no pretrained weights)", "chemprop", None),
        Foundation("chemeleon", "Chemprop D-MPNN started from the CheMeleon pretrained encoder", "chemprop", "chemeleon", _chemeleon_weights),
        Foundation("unimol", f"Uni-Mol v2 {UNIMOL_SIZE} 3D encoder", "unimol", "unimol", _unimol_weights),
        Foundation("atom_jepa", "Atom-JEPA EquiformerV3 encoder, pretrained on 19M molecules", "atom_jepa", "atom_jepa", _atom_jepa_weights("molecules")),
        Foundation("atom_jepa_bio", "Atom-JEPA continued on ~1.5M PCBA bioassay molecules", "atom_jepa_bio", "atom_jepa_bio", _atom_jepa_weights("bio")),
        Foundation("monroe", "Monroe GRIT 3D encoder (embedding only)", None, "monroe", _monroe_weights, trainable=False),
    )
}
TRAINABLE = [f.name for f in CATALOG.values() if f.trainable]
PRETRAINED = tuple(f.name for f in CATALOG.values() if f.pretrained)


# --------------------------------------------------------------------------------------------------------------
# Fine-tunes on disk
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Finetune:
    """One fine-tuned weights file."""

    model: str  # a CATALOG name
    name: str  # the model directory or run folder it belongs to
    run: str  # the run inside it (seed, learning rate, stage), "" when the folder holds one
    weights: Path
    alias: str

    @property
    def kind(self) -> Kind:
        return KINDS[CATALOG[self.model].kind]

    @property
    def spec(self) -> str:
        return self.kind.spec(self.weights)


def _short_names(models: list[tuple[str, list[Path]]]) -> list[str]:
    """Model names without the path prefix shared by all models."""
    shared = len(Path(os.path.commonpath([str(Path(name).parent) for name, _ in models])).parts)
    return ["_".join(Path(name).parts[shared:]) for name, _ in models]


def _assign_aliases(kind: Kind, models: list[tuple[str, list[Path]]]) -> dict[str, str]:
    """Alias every run's weights path that has none yet, e.g. chemprop challenge seed0 -> challenge_0.

    The path prefix shared by all models is dropped, and the model name is truncated to _MODEL_CHARS characters.
    Models that truncate to the same name get a numeric suffix from ensure_aliases.
    """
    entries = []
    for short, (_, weights) in zip(_short_names(models), models):
        model = short[:_MODEL_CHARS].rstrip("_+.-")
        entries.extend((w, kind.alias_for(model, w)) for w in weights)
    return ensure_aliases(entries)


def _home_finetunes() -> list[Finetune]:
    """Fine-tunes in the per-kind homes (results/chemprop/<model>/seed<N>/..., results/unimol/<model>/lr_*_seed_*/...)."""
    found = []
    for kind in KINDS.values():
        home = kind.home()
        models = list_models(home, kind) if home.is_dir() else []
        if not models:
            continue
        aliases = _assign_aliases(kind, models)  # from all runs, so what's on disk never changes a run's alias
        for name, weights in models:
            model = "chemeleon" if kind.name == "chemprop" and "chemeleon" in Path(name).parts else kind.name
            for w in weights:
                if w.is_file():
                    found.append(Finetune(model, name, kind.run_dir(w).name, w, aliases[str(w.resolve())]))
    return found


_TRAIN_RUN = re.compile(r"^(?P<id>\d+)__train__model=(?P<model>[a-z_]+?)(?:_(?P<rest>.*))?$")


def _run_finetunes(runs_dir: Path) -> list[Finetune]:
    """Fine-tunes in `fm train` run folders: `<id>__train__model=<model>_<flags>/.../<weights>`."""
    if not runs_dir.is_dir():
        return []
    found: list[tuple[Finetune, str]] = []
    for folder in sorted(runs_dir.iterdir()):
        match = _TRAIN_RUN.fullmatch(folder.name)
        if not folder.is_dir() or not match or match["model"] not in CATALOG or CATALOG[match["model"]].kind is None:
            continue
        model = match["model"]
        kind = KINDS[CATALOG[model].kind]
        weights = sorted(w for w in folder.rglob(kind.weights_glob) if w.is_file())
        name = f"{match['id']} {match['rest'] or ''}".strip()
        for w in weights:
            run_dir = w.parent.relative_to(folder)
            run = "" if len(weights) == 1 else str(run_dir)
            suffix = ""
            if len(weights) > 1:
                pattern = next((p for p in reversed(run_dir.parts) if kind.run_pattern.fullmatch(p)), None)
                suffix = "_" + (kind.alias_suffix.format(**kind.run_pattern.fullmatch(pattern).groupdict()) if pattern else run.replace("/", "_"))
            found.append((Finetune(model, name, run, w, ""), f"{model}_{match['id']}{suffix}"))
    aliases = ensure_aliases([(f.weights, suggestion) for f, suggestion in found])
    return [Finetune(f.model, f.name, f.run, f.weights, aliases[str(f.weights.resolve())]) for f, _ in found]


def discover(runs_dir: Path | None = None) -> list[Finetune]:
    """Every fine-tuned weights file on this machine: `fm train` run folders first, then the per-kind homes."""
    return _run_finetunes(runs_dir or RUNS_DIR) + _home_finetunes()


# --------------------------------------------------------------------------------------------------------------
# Sizes and embedding status
# --------------------------------------------------------------------------------------------------------------


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError


def _bytes(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())  # stat follows the hub's symlinks to the blobs


class Coverage:
    """How much of a molecule set each cached embedding holds. The set is loaded once, on first use."""

    def __init__(self, cache_dir: Path, molecules: Callable[[], pl.DataFrame], label: str) -> None:
        self.cache_dir = cache_dir
        self._molecules = molecules
        self.label = label

    @cache
    def smiles(self) -> set[str] | None:
        try:
            return set(self._molecules()["SMILES"].to_list())
        except (OSError, ValueError, pl.exceptions.PolarsError):
            return None  # no data on this machine: report what is cached without a denominator

    def stem(self, spec: str, kind: Kind | None = None, stem: str | None = None) -> str | None:
        """The cache stem holding a spec's rows, or None. Weights that are gone fall back to their one cached embedding."""
        if kind is None:
            return stem
        try:
            return kind.cache_name(spec)
        except OSError:
            cached = sorted(p.stem for p in self.cache_dir.glob(f"{kind.cache_prefix(spec)}_*.idx"))
            return cached[0] if len(cached) == 1 else None

    def cell(self, stem: str | None) -> str:
        idx = self.cache_dir / f"{stem}.idx" if stem else None
        if idx is None or not idx.is_file() or not idx.with_suffix(".npy").is_file():
            return "[dim]none[/]"
        indexed = idx.read_text().splitlines()
        wanted = self.smiles()
        if wanted is None:
            return f"{len(indexed)} mols"
        have = len(wanted.intersection(indexed))
        colour = "green" if have == len(wanted) else "yellow" if have else "red"
        return f"[{colour}]{have}/{len(wanted)}[/]"


def _train_test(data_dir: Path) -> Callable[[], pl.DataFrame]:
    def load() -> pl.DataFrame:
        from data_tools.representations_cli import _molecules

        return _molecules(data_dir)

    return load


# --------------------------------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _width(cell: str) -> int:
    return Text.from_markup(cell).cell_len


@dataclass(frozen=True)
class Column:
    header: str
    justify: str = "left"
    fold: int = 0  # >0: may wrap, and counts this many cells wide when deciding what fits (it wraps past it)
    style: str | None = None


def _render(columns: list[Column], rows: list[list[str]], drop: list[str], keep: set[str] = frozenset()) -> Table:
    """A table of the columns that fit the terminal, dropping whole columns in `drop` order until the rest fit.

    Columns in `keep` stay regardless. A dropped column is gone rather than squeezed, so every cell shown is whole:
    an alias that is cut short names nothing.
    """
    shown = list(range(len(columns)))

    def needed() -> int:
        widths = []
        for i in shown:
            full = max(_width(c) for c in [columns[i].header, *(r[i] for r in rows)])
            widths.append(min(full, columns[i].fold) if columns[i].fold else full)
        return sum(widths) + 2 * (len(widths) - 1)

    for header in drop:
        if needed() <= console.width:
            break
        index = next(i for i, c in enumerate(columns) if c.header == header)
        if columns[index].header not in keep:
            shown.remove(index)
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", pad_edge=False)
    for i in shown:
        c = columns[i]
        table.add_column(c.header, justify=c.justify, style=c.style, no_wrap=not c.fold, overflow="fold" if c.fold else "ellipsis",
                         min_width=None if c.fold else max(_width(x) for x in [c.header, *(r[i] for r in rows)]),
                         max_width=c.fold if 0 < c.fold < console.width else None)
    for row in rows:
        table.add_row(*(row[i] for i in shown))
    return table


def _avail(args: argparse.Namespace) -> None:
    coverage = Coverage(args.cache_dir, _train_test(args.data_dir), "train+test")
    finetunes = discover()
    rows = []
    for f in CATALOG.values():
        if f.pretrained_weights is None:
            pretrained, embedded = "[dim]none[/]", "[dim]-[/]"
        else:
            files = [p for p in f.pretrained_weights() if p.is_file()]
            pretrained = _size(_bytes(files)) if files else "[dim]not downloaded[/]"
            embedded = coverage.cell(f.pretrained_stem)
        mine = [t for t in finetunes if t.model == f.name]
        tuned = f"{len(mine)} · {_size(_bytes([t.weights for t in mine]))}" if mine else ("[dim]0[/]" if f.trainable else "[dim]-[/]")
        rows.append([f.name, pretrained, embedded, tuned, "[green]✓[/]" if f.trainable else "[dim]-[/]", f.about])
    columns = [
        Column("Model", style="bold"),
        Column("Pretrained", "right"),
        Column("Embedded", "right"),
        Column("Fine-tunes", "right"),
        Column("Train", "center"),
        Column("About", fold=30),
    ]
    console.print(_render(columns, rows, drop=["About", "Train"]))
    console.print(
        f"[dim]Pretrained = encoder weights on disk (downloaded on first use). Embedded = its cached molecules out of the "
        f"{coverage.label} set in {args.cache_dir}. `fm finetunes` lists the fine-tunes; `fm train --model NAME` makes one.[/]"
    )


def _finetunes(args: argparse.Namespace) -> None:
    coverage = Coverage(args.cache_dir, _train_test(args.data_dir), "train+test")
    found = [f for f in discover() if not args.models or f.model in args.models]
    if not found:
        wanted = f" for {', '.join(args.models)}" if args.models else ""
        print(f"No fine-tuned weights{wanted} in {RUNS_DIR} or " + ", ".join(str(k.home()) for k in KINDS.values()), file=sys.stderr)
        sys.exit(1)

    columns = [
        Column("Model", style="bold"),
        Column("Alias"),
        Column("Size", "right"),
        Column("Embedded", "right"),
        Column("Run"),
        Column("Fine-tune", fold=24),
        Column("Path", fold=1 << 30),  # counted at full width: shown only when it fits whole
    ]
    rows = [
        [
            f.model,
            escape(f.alias),
            _size(f.weights.stat().st_size),
            coverage.cell(coverage.stem(f.spec, f.kind)),
            escape(f.run),
            escape(f.name),
            escape(_display_path(f.weights)),
        ]
        for f in found
    ]
    if not any(r[4] for r in rows):  # no folder holds several runs
        columns.pop(4)
        rows = [r[:4] + r[5:] for r in rows]
    if args.no_paths:
        columns.pop()
        rows = [r[:-1] for r in rows]
    show = {"Path"} if args.paths else set()
    table = _render(columns, rows, drop=["Path", "Fine-tune", "Run"], keep=show)
    console.print(table)
    shown = {c.header for c in table.columns}
    note = f"Embedded = cached molecules out of the {coverage.label} set in {args.cache_dir}. "
    note += "Use an alias as chemprop[ALIAS], unimol[ALIAS], atom_jepa[ALIAS] wherever an input is accepted"
    note += "" if "Path" in shown or args.no_paths else "; --paths shows the weights paths"
    console.print(f"[dim]{note}.[/]")


def _dataset_molecules(references: list[str], data_dir: Path) -> pl.DataFrame:
    from data_tools.describe import resolve_table

    frames = []
    for reference in references:
        df, name = resolve_table(reference, data_dir)
        print(f"dataset {name}: {df.height} rows")
        frames.append(df.select("SMILES"))
    molecules = standardization.standardize_column(pl.concat(frames), "SMILES")
    return molecules.unique(subset=["SMILES"], keep="first", maintain_order=True)


def _embed_targets(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The (label, input spec) pairs `--model` and `--only` select."""
    finetunes = discover()
    by_alias = {f.alias: f for f in finetunes}
    targets: list[tuple[str, str]] = []
    for name in args.model or list(CATALOG):
        if name in by_alias:
            f = by_alias[name]
            targets.append((f"{f.model} {f.alias}", f.spec))
            continue
        if name not in CATALOG:
            raise SystemExit(f"fm embed: {name!r} is neither a foundation model ({', '.join(CATALOG)}) nor a fine-tune alias (see `fm finetunes`)")
        foundation = CATALOG[name]
        if args.only != "finetuned" and foundation.pretrained:
            targets.append((f"{name} pretrained", foundation.pretrained))
        if args.only != "pretrained":
            targets.extend((f"{name} {f.alias}", f.spec) for f in finetunes if f.model == name)
    return list(dict.fromkeys(targets))


def _embed(args: argparse.Namespace) -> None:
    from data_tools.inputs import featurize

    targets = _embed_targets(args)
    if not targets:
        raise SystemExit("fm embed: nothing to embed (no pretrained encoder or fine-tune matches)")
    if args.dataset:
        molecules = _dataset_molecules(args.dataset, args.data_dir)
        label = " + ".join(args.dataset)
    else:
        molecules = _train_test(args.data_dir)()
        label = "train+test"
    print(f"Embedding {len(molecules)} unique molecules ({label}) with {len(targets)} model(s) into {args.cache_dir}")
    failed = []
    for index, (name, spec) in enumerate(targets, 1):
        print(f"\n[{index}/{len(targets)}] {name}")
        try:
            featurize(molecules, [spec], args.cache_dir)
        except Exception as error:  # one broken model shouldn't stop the rest of a long GPU job
            print(f"{name}: failed: {type(error).__name__}: {error}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"\n{len(failed)}/{len(targets)} failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing train.csv and test.csv")
    parser.add_argument("--cache-dir", type=Path, default=FEATURE_DIR, help="Directory for cached feature matrices")
    standardization.add_argument(parser)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="{avail,finetunes,embed,train}")

    avail = commands.add_parser("avail", help="List every foundation model, its pretrained weights and fine-tunes, with sizes")
    _common(avail)
    avail.set_defaults(run=_avail)

    finetunes = commands.add_parser("finetunes", help="List the fine-tuned weights on this machine: alias, size, embedding status, path")
    finetunes.add_argument("--models", nargs="+", choices=TRAINABLE, metavar="MODEL", help=f"Only these models ({', '.join(TRAINABLE)})")
    shown = finetunes.add_mutually_exclusive_group()
    shown.add_argument("--paths", action="store_true", help="Always show the weights path, even when the terminal is narrow")
    shown.add_argument("--no-paths", action="store_true", help="Never show the weights path (the alias stands in for it)")
    _common(finetunes)
    finetunes.set_defaults(run=_finetunes)

    embed = commands.add_parser(
        "embed",
        help="Embed molecules with pretrained encoders and fine-tunes, caching the features (already-cached molecules are skipped)",
    )
    embed.add_argument(
        "--model",
        nargs="+",
        metavar="MODEL",
        help=f"Foundation models ({', '.join(CATALOG)}: the pretrained encoder and every fine-tune of it) or fine-tune "
        "aliases from `fm finetunes` (that one run). Default: all of them",
    )
    embed.add_argument(
        "--only",
        choices=["pretrained", "finetuned"],
        help="With a model name, embed only its pretrained encoder, or only its fine-tunes",
    )
    embed.add_argument(
        "--dataset",
        nargs="+",
        metavar="DATASET",
        help="Datasets (name, alias, or CSV path with a SMILES column) to embed. Default: the challenge train and test sets",
    )
    _common(embed)
    embed.set_defaults(run=_embed)

    commands.add_parser("train", add_help=False, help="Fine-tune or train a model (see `train --model NAME --help`)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for `foundation-model` and `fm`."""
    argv = sys.argv[1:] if argv is None else argv
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "foundation-model"
    prog = prog if prog in ("fm", "foundation-model") else "foundation-model"
    if argv and argv[0] == "train":
        # Training owns its own parser (each backend adds flags, chemprop's pass through), so hand it the rest.
        from models import train_model

        train_model.main(argv[1:], prog=f"{prog} train")
        return
    with clean_interrupt(prog):
        args = build_parser(prog).parse_args(argv)
        standardization.set_mode(args.standardization)
        args.run(args)


if __name__ == "__main__":
    main()
