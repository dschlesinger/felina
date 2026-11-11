"""TDC Obach drug half-life: continuous label `Y`, 667 molecules."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_half_life_obach/tdc_half_life_obach.csv"
FILE = "metabolism/tdc_half_life_obach.csv"

SPEC = DatasetSpec(
    name="tdc_half_life_obach",
    group="metabolism",
    description="TDC Obach drug half-life, regression (667)",
    order=140,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_half_life_obach": Source(
            kind="external",
            heads=[(r"^(Y)$", "HalfLife-Obach")],
            log10p1=True,
            description="TDC Obach half-life, log10(x+1)",
        ),
    },
)
