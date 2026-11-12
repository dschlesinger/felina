"""CLI for managing trained foundation models (chemprop runs and fine-tuned Uni-Mol and Atom-JEPA runs)."""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from data_tools.inputs import featurize
from data_tools.representations_cli import _molecules
from representations.chemprop_weights import cache_name as chemprop_cache_name
from representations.foundation_alias import ensure_aliases
from representations.unimol import cache_name as unimol_cache_name
from representations.atom_jepa import cache_name as atom_jepa_cache_name
from data_tools.paths import DATA_DIR, FEATURE_DIR, RESULTS_DIR

_MODEL_CHARS = 10

console = Console()


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
    cache_name: Callable[[str], str]

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


KINDS = {
    "chemprop": Kind(
        "chemprop",
        RESULTS_DIR / "chemprop",
        "FOUNDATION_MODEL_HOME",
        re.compile(r"seed(?P<seed>\d+)"),
        "{seed}",
        Path("model_0/best.pt"),
        chemprop_cache_name,
    ),
    "unimol": Kind(
        "unimol",
        RESULTS_DIR / "unimol",
        "UNIMOL_MODEL_HOME",
        re.compile(r"lr_(?P<lr>[0-9.]+(?:e[+-]?\d+)?)_seed_(?P<seed>\d+)"),
        "lr{lr}_s{seed}",
        Path("model_0.pth"),
        unimol_cache_name,
    ),
    **{
        name: Kind(
            name,
            RESULTS_DIR / name,
            f"{name.upper()}_MODEL_HOME",
            re.compile(r"lr_(?P<lr>[0-9.]+(?:e[+-]?\d+)?)_seed_(?P<seed>\d+)"),
            "lr{lr}_s{seed}",
            Path("encoder.pt"),
            atom_jepa_cache_name,
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


def _short_names(models: list[tuple[str, list[Path]]]) -> list[str]:
    """Model names without the path prefix shared by all models."""
    shared = len(Path(os.path.commonpath([str(Path(name).parent) for name, _ in models])).parts)
    return ["_".join(Path(name).parts[shared:]) for name, _ in models]


def _print_models(kind: Kind, home: Path, models: list[tuple[str, list[Path]]], aliases: dict[str, str]) -> None:
    table = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", title=f"{kind.name}: {home}", title_style="bold")
    table.add_column("Model", style="bold", no_wrap=True, max_width=_MODEL_CHARS, overflow="ellipsis")
    table.add_column("Seeds" if kind.name == "chemprop" else "Runs", justify="right")
    table.add_column("Aliases", overflow="fold")
    for name, (_, weights) in zip(_short_names(models), models):
        trained = sum(w.is_file() for w in weights)
        runs = str(len(weights)) if trained == len(weights) else f"[yellow]{trained}/{len(weights)}[/] [dim]trained[/]"
        table.add_row(name, runs, escape(", ".join(aliases[str(w.resolve())] for w in weights)))
    console.print(table)


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


def _print_paths(kind: Kind, models: list[tuple[str, list[Path]]], aliases: dict[str, str]) -> None:
    """Print each model's per-run weights paths and aliases, ready to paste into `--input chemprop[...]` or `unimol[...]`."""
    for name, weights in models:
        print(name)
        for w in weights:
            print(f"  {w}  [{aliases[str(w.resolve())]}]" + ("" if w.is_file() else "  (missing)"))


def _generate_caches(kind: Kind, models: list[tuple[str, list[Path]]], data_dir: Path, cache_dir: Path) -> int:
    """Embed the train+test molecules with every run's weights that has no feature cache yet.

    Runs without weights are skipped; a run whose cache files exist is left untouched. Returns how many were generated.
    """
    molecules = None
    generated = 0
    for name, weights in models:
        for path in weights:
            label = f"{kind.name}: {name}/{kind.run_dir(path).name}"
            if not path.is_file():
                print(f"{label}: no weights, skipped")
                continue
            spec = kind.spec(path)
            stem = cache_dir / kind.cache_name(spec)
            if stem.with_suffix(".npy").exists() and stem.with_suffix(".idx").exists():
                print(f"{label}: cached")
                continue
            if molecules is None:
                molecules = _molecules(data_dir)
            print(f"{label}: generating cache for {len(molecules)} molecules")
            featurize(molecules, [spec], cache_dir)
            generated += 1
    return generated


# `--type` words that narrow both the fine-tuned runs and the pretrained models to those whose name has this part
# (a run's name is its path under the model home: chemeleon runs live under results/chemprop/chemeleon/, and the
# pretrained model is `chemeleon`). They combine with the others: `--type finetuned chemeleon`.
FILTERS: dict[str, Callable[[str], bool]] = {"chemeleon": lambda name: "chemeleon" in Path(name).parts}
_GROUPS = ("all", "finetuned", "pretrained")
PRETRAINED = ("chemeleon", "unimol", "atom_jepa", "atom_jepa_bio")  # registered inputs that are foundation models without a training run


def _generate_pretrained(names: list[str], data_dir: Path, cache_dir: Path) -> int:
    """Embed the train+test molecules with each named pretrained (not fine-tuned) model that has no feature cache yet."""
    molecules = None
    generated = 0
    for name in names:
        stem = cache_dir / name
        if stem.with_suffix(".npy").exists() and stem.with_suffix(".idx").exists():
            print(f"pretrained: {name}: cached")
            continue
        if molecules is None:
            molecules = _molecules(data_dir)
        print(f"pretrained: {name}: generating cache for {len(molecules)} molecules")
        featurize(molecules, [name], cache_dir)
        generated += 1
    return generated


def main() -> None:
    """Entry point for the foundation-model CLI."""
    parser = argparse.ArgumentParser(
        description="Manage trained foundation models. Looks in "
        + ", ".join(f"${k.home_env} for {k.name} (default {k.default_home})" for k in KINDS.values())
        + ".",
    )
    parser.add_argument("--list", action="store_true", help="List the available models with their run counts")
    parser.add_argument("--paths", action="store_true", help="Show each model's per-run weights path instead of the run count")
    parser.add_argument(
        "--generate-cache",
        action="store_true",
        help="Generate the feature cache (train+test molecules) for every run that doesn't have one, and for the pretrained "
        + ", ".join(PRETRAINED),
    )
    parser.add_argument(
        "--type",
        nargs="+",
        choices=[*_GROUPS, *KINDS, *FILTERS],
        default=["all"],
        help="What to manage, any combination: all (default), finetuned (every trained run), pretrained (the non-fine-tuned "
        + ", ".join(PRETRAINED)
        + ", with --generate-cache), a kind ("
        + ", ".join(KINDS)
        + "), and filters ("
        + ", ".join(FILTERS)
        + ") that keep only the matching models, e.g. --type finetuned chemeleon (a filter alone covers both fine-tuned and pretrained)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing train.csv and test.csv")
    parser.add_argument("--cache-dir", type=Path, default=FEATURE_DIR, help="Directory for cached feature matrices")
    args = parser.parse_args()

    if not (args.list or args.paths or args.generate_cache):
        parser.print_help()
        return

    filters = [FILTERS[t] for t in args.type if t in FILTERS]
    explicit = [t for t in args.type if t in KINDS]
    only_filters = all(t in FILTERS for t in args.type)  # a filter alone covers fine-tuned and pretrained
    every_kind = "all" in args.type or "finetuned" in args.type or only_filters
    kinds = list(KINDS.values()) if every_kind else [KINDS[t] for t in explicit]
    pretrained = [n for n in PRETRAINED if all(f(n) for f in filters)] if "all" in args.type or "pretrained" in args.type or only_filters else []
    if not kinds and not args.generate_cache:
        parser.error("--type pretrained only applies with --generate-cache")
    found = False
    generated = 0
    for kind in kinds:
        home = kind.home()
        every_model = list_models(home, kind) if home.is_dir() else []
        models = [(name, weights) for name, weights in every_model if all(f(name) for f in filters)]
        if not models:
            if explicit and not filters:
                reason = "does not exist" if not home.is_dir() else "holds no models"
                print(f"{kind.name} model home {home} {reason} (set ${kind.home_env} to change it)", file=sys.stderr)
                sys.exit(1)
            continue
        found = True
        aliases = _assign_aliases(kind, every_model)  # from all runs, so a filter never changes a run's alias
        if args.generate_cache:
            generated += _generate_caches(kind, models, args.data_dir, args.cache_dir)
        elif args.paths:
            _print_paths(kind, models, aliases)
        else:
            _print_models(kind, home, models, aliases)
    if args.generate_cache and pretrained:
        found = True  # pretrained models need no run directory
        generated += _generate_pretrained(pretrained, args.data_dir, args.cache_dir)
    if not found:
        print("No models found in " + ", ".join(str(k.home()) for k in kinds), file=sys.stderr)
        sys.exit(1)
    if args.generate_cache:
        print(f"Generated {generated} cache(s) in {args.cache_dir}")


if __name__ == "__main__":
    main()
