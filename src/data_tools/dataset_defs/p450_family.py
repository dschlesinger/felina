"""P450-family affinity data from ChEMBL and BindingDB, one head per protein per source.

Fetch walks every reviewed UniProt entry in Pfam PF00067, pulls each protein's ligand affinities, and writes one
CSV per protein under `families/` plus a manifest. Derive pivots the manifest into `family_heads.csv`, one
`{gene}_pIC50_{source}` column per protein per source (see `data_tools.family_heads` for why pooling them into a
single `pAffinity` column does not work).

The fetch is the expensive one in the project: it queries UniProt, ChEMBL, and BindingDB for hundreds of
proteins. `data fetch p450_family` re-derives from the manifest without re-crawling; pass `--redownload` to crawl
again.

Fetch casts wide (200 proteins, 5+ records each) so the manifest holds more than any one run needs; derive is
where the selection happens, so narrowing the heads is a re-derive, not a refetch.
"""

from pathlib import Path

from data_tools.dataset_spec import DatasetSpec, Source

DIR = "families"
MANIFEST = f"{DIR}/manifest.csv"
FILE = "family_heads.csv"

PFAM_ID = "PF00067"  # the cytochrome P450 Pfam family
BINDINGDB_CUTOFF_NM = 10000.0  # ignore affinities weaker than 10 uM
MAX_WORKERS = 8  # parallel target fetches
FETCH_TOP_N = 200  # proteins to write a CSV for, richest first
FETCH_MIN_RECORDS = 5  # skip proteins with less data than this

HEAD_ORGANISM = "Homo sapiens"  # most PF00067 entries are bacterial or plant P450s
HEAD_MIN_RECORDS = 100
HEAD_TOP_N = 20
HEAD_AFFINITY_TYPES = ("IC50",)  # the endpoint the challenge scores; mixing in Ki/Kd/EC50 re-pools assays


def _fetch(data_dir: Path) -> None:
    from data_tools.family_datasets import collect_family_targets, write_datasets

    targets = collect_family_targets(PFAM_ID, BINDINGDB_CUTOFF_NM, MAX_WORKERS)
    write_datasets(targets, data_dir / DIR, FETCH_TOP_N, FETCH_MIN_RECORDS)


def _derive(data_dir: Path) -> None:
    from data_tools.family_heads import write_heads

    write_heads(
        data_dir / MANIFEST,
        data_dir / FILE,
        organism=HEAD_ORGANISM,
        min_records=HEAD_MIN_RECORDS,
        top_n=HEAD_TOP_N,
        affinity_types=HEAD_AFFINITY_TYPES,
    )


SPEC = DatasetSpec(
    name="p450_family",
    group="auxiliary",
    description="ChEMBL + BindingDB affinity across the P450 family, pivoted into per-protein heads",
    order=70,
    fetched=(MANIFEST,),
    fetch=_fetch,
    derived=(FILE,),
    derive=_derive,
    directories=(DIR,),
    sources={
        "chembl": Source(
            kind="external",
            heads=[(r"^(.+)_pIC50_chembl$", "{}-ChEMBL")],
            description="ChEMBL pIC50 across the P450 family",
        ),
        "bindingdb": Source(
            kind="external",
            heads=[(r"^(.+)_pIC50_bindingdb$", "{}-BindingDB")],
            description="BindingDB pIC50 across the P450 family",
        ),
    },
)
