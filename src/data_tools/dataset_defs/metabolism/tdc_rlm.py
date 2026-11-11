"""TDC rat liver microsomal stability: binary label `Y`, 5,590 molecules."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_rlm/tdc_rlm.csv"
FILE = "metabolism/tdc_rlm.csv"

SPEC = DatasetSpec(
    name="tdc_rlm",
    group="metabolism",
    description="TDC rat liver microsomal stability, binary (5,590)",
    order=120,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_rlm": Source(
            kind="external",
            heads=[(r"^(Y)$", "RLM-TDC")],
            task="classification",
            description="TDC rat liver microsomal stability, binary",
        ),
    },
)
