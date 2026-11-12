"""The shape of a dataset definition.

A `DatasetSpec` is one dataset's complete recipe, and each dataset gets its own module under
`data_tools.dataset_defs` holding nothing but that recipe: its source URL, how to fetch it, how to derive the
model-ready table, and which training heads the result offers. `data_tools.datasets.registry()` discovers those
modules, so adding a dataset means adding one file and changing nothing else.

Two halves:

  obtaining    `fetch` pulls raw data from the source; `derive` (optional) builds the model-ready table from what
               fetch wrote. Both take the data directory and write under it. `fetched` and `derived` list what each
               writes, which is what `data list` reports on and `data remove` deletes.
  training     `sources` declares the heads the resulting file offers, keyed by the name `--datasets NAME:ISO`
               selects. A dataset may declare none (nothing to train on, like the blinded test split), one, or
               several from a single file (`family_heads.csv` yields both `chembl` and `bindingdb`).
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

    `heads` pairs a regex whose single group captures the isoform or gene with a template for the head name, so
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

    def __post_init__(self) -> None:
        if not self.fetched:
            raise ValueError(f"dataset {self.name!r}: fetched must name at least one file")
        if self.derived and self.derive is None:
            raise ValueError(f"dataset {self.name!r}: derived files are listed but there is no derive step")
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
