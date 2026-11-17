"""The shape of a dataset definition.

A `DatasetSpec` is one dataset's complete recipe, and each dataset gets its own module under
`data_tools.dataset_defs` holding nothing but that recipe: its source URL, how to fetch it, how to derive the
model-ready table, and which training heads the result offers. `data_tools.datasets.registry()` discovers those
modules, so adding a dataset means adding one file and changing nothing else.

Three halves:

  obtaining    `fetch` pulls raw data from the source; `derive` (optional) builds the model-ready table from what
               fetch wrote. Both take the data directory and write under it. `fetched` and `derived` list what each
               writes, which is what `data list` reports on and `data remove` deletes.
  training     `sources` declares the heads the resulting file offers, keyed by the name `--datasets NAME:TARGET`
               selects. A dataset may declare none (nothing to train on, like the blinded test split), one, or
               several from a single file (`family_heads.csv` yields both `chembl` and `bindingdb`).
  evaluation   `scored` marks the one dataset a challenge is scored on. It carries the target columns, the
               molecule and identifier column names, the suffix stripped for display, and the name of the
               dataset holding its blinded prediction set. `datasets.challenge()` returns it, and every module
               that used to hold its own copy of the CYP target list now asks for it instead. Switching
               challenge means adding a scored dataset module and pointing `CHALLENGE_DATASET` at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import polars as pl

Step = Callable[[Path], None]


@dataclass(frozen=True)
class Source:
    """One selectable training source: how a dataset's file turns into model heads.

    `heads` pairs a regex whose single group captures the target or gene with a template for the head name, so
    `(r"^(.+)_pIC50_TDI_condition$", "{}-TDI")` turns `CYP2D6_pIC50_TDI_condition` into the head `CYP2D6-TDI`.
    """

    kind: str  # "inhouse" (keyed on Molecule_Name) or "external" (joined on the InChIKey block)
    heads: list[tuple[str, str]]
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    description: str = ""
    task: str = "regression"  # "classification" for 0/1 labels
    log10p1: bool = False  # label spans orders of magnitude: train on log10(x + 1)
    filename: str = ""  # file to read, relative to the data directory; defaults to the dataset's primary file


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset: where it comes from, what it writes, and what it trains as."""

    name: str
    group: str
    description: str
    fetched: tuple[str, ...]  # files fetch writes, relative to the data directory
    fetch: Step
    derived: tuple[str, ...] = ()  # files derive writes, relative to the data directory
    derive: Step | None = None
    directories: tuple[str, ...] = ()  # extra directories fetch fills, removed whole by `data remove`
    sources: dict[str, Source] = field(default_factory=dict)  # training sources, by `--datasets` name
    order: int = 1000  # listing order; ties break on name

    # --- evaluation: set on the dataset a challenge is scored on -------------------------------------------
    scored: bool = False  # this is a challenge's scored dataset; `datasets.challenge()` finds it
    targets: tuple[str, ...] = ()  # scored label columns, in reporting order; required when scored
    label_suffix: str = ""  # trimmed off a target for display, e.g. "_pIC50_direct_inhibition"
    smiles_column: str = "SMILES"
    id_column: str = "Molecule_Name"
    blind: str = ""  # name of the dataset holding the blinded molecules to predict

    def __post_init__(self) -> None:
        if not self.fetched:
            raise ValueError(f"dataset {self.name!r}: fetched must name at least one file")
        if self.derived and self.derive is None:
            raise ValueError(f"dataset {self.name!r}: derived files are listed but there is no derive step")
        if self.scored and not self.targets:
            raise ValueError(f"dataset {self.name!r}: a scored dataset must name its targets")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"dataset {self.name!r}: targets repeats a column")
        # A source reads its dataset's model-ready file unless it names another.
        object.__setattr__(
            self,
            "sources",
            {n: s if s.filename else replace(s, filename=self.primary) for n, s in self.sources.items()},
        )

    @property
    def files(self) -> tuple[str, ...]:
        return self.fetched + self.derived

    @property
    def primary(self) -> str:
        """The file models consume: the last derived file, else the last fetched one."""
        return self.files[-1]

    def shorthand(self, target: str) -> str:
        """The short name a target column reports under: the column with `label_suffix` trimmed off."""
        return target.removesuffix(self.label_suffix) if self.label_suffix else target

    def resolve_targets(self, selected: list[str] | tuple[str, ...] | None) -> list[str]:
        """Expand `--dataset NAME:A,B` selectors into full target columns, in the spec's own order.

        Each selector is a full column name, the shorthand `shorthand()` reports, or any unambiguous tail of
        one — so a CYP challenge takes `CYP3A4_pIC50_direct_inhibition`, `CYP3A4` and the bare `3A4` alike.
        Matching ignores case. `None` or an empty selection means every target. Raises ValueError naming the
        unknown selector and the valid choices.
        """
        if not selected:
            return list(self.targets)
        by_name = {t.casefold(): t for t in self.targets}
        by_name |= {self.shorthand(t).casefold(): t for t in self.targets}

        def resolve(selector: str) -> str | None:
            key = selector.casefold()
            if key in by_name:
                return by_name[key]
            # A tail of a shorthand, accepted only while it names exactly one target (`3A4` -> `CYP3A4`).
            tails = {t for t in self.targets if self.shorthand(t).casefold().endswith(key)}
            return tails.pop() if len(tails) == 1 else None

        resolved = {s: resolve(s) for s in selected}
        unknown = [s for s, t in resolved.items() if t is None]
        if unknown:
            raise ValueError(
                f"dataset {self.name!r} has no target(s) {', '.join(unknown)}; "
                f"choose from {', '.join(self.shorthand(t) for t in self.targets)}"
            )
        wanted = set(resolved.values())
        return [t for t in self.targets if t in wanted]  # spec order, deduplicated
