"""PubChem AID 1851 (Veith CYP qHTS panel): five isoforms, potency and percent inhibition.

Why this source and how its columns are built is documented in `data_tools.pubchem`; the short version is that
the percent-inhibition column carries the 42,460 measured negatives a potency-only ingest would discard.

Fetched raw (~24 MB) and reshaped by the derive step, so changing the reshape and re-running `data fetch
pubchem_aid1851` rebuilds the heads without touching the network.
"""

from pathlib import Path

from data_tools.dataset_spec import DatasetSpec, Source

RAW = "pubchem_aid1851_raw.csv"
FILE = "pubchem_aid1851.csv"


def _fetch(data_dir: Path) -> None:
    from data_tools.pubchem import download_raw

    download_raw(data_dir / RAW)


def _derive(data_dir: Path) -> None:
    from data_tools.pubchem import reshape, write_csv

    fieldnames, rows = reshape(data_dir / RAW)
    write_csv(fieldnames, rows, data_dir / FILE)


SPEC = DatasetSpec(
    name="pubchem_aid1851",
    group="auxiliary",
    description="PubChem AID 1851 (Veith CYP qHTS panel) reshaped into per-isoform heads",
    order=60,
    fetched=(RAW,),
    fetch=_fetch,
    derived=(FILE,),
    derive=_derive,
    sources={
        "pubchem": Source(
            kind="external",
            heads=[
                (r"^(.+)_pIC50_aid1851$", "{}-PubChem"),
                (r"^(.+)_pctinhib_aid1851$", "{}-PubChem-pctinhib"),
            ],
            description="PubChem AID 1851 qHTS panel: pIC50 and percent inhibition",
        ),
    },
)
