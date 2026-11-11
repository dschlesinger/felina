"""ExpansionRx challenge train data: microsomal clearance plus physchem endpoints (CC-BY-4.0).

Only the two CLint columns become heads. LogD, KSOL, permeability, and PPB are in the file and available to read,
but they measure something other than metabolism and are not wired up as heads here.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv"
FILE = "metabolism/expansionrx.csv"

SPEC = DatasetSpec(
    name="expansionrx",
    group="metabolism",
    description="ExpansionRx challenge train: HLM/MLM CLint plus LogD, KSOL, permeability, PPB (CC-BY-4.0)",
    order=100,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "expansionrx": Source(
            kind="external",
            heads=[(r"^(HLM|MLM) CLint$", "{}-ExpansionRx")],
            log10p1=True,
            description="ExpansionRx microsomal CLint (human, mouse), log10(x+1)",
        ),
    },
)
