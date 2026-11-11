"""TDC human liver microsomal stability: binary label `Y`, 6,013 molecules."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_hlm/tdc_hlm.csv"
FILE = "metabolism/tdc_hlm.csv"

SPEC = DatasetSpec(
    name="tdc_hlm",
    group="metabolism",
    description="TDC human liver microsomal stability, binary (6,013)",
    order=110,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_hlm": Source(
            kind="external",
            heads=[(r"^(Y)$", "HLM-TDC")],
            task="classification",
            description="TDC human liver microsomal stability, binary",
        ),
    },
)
