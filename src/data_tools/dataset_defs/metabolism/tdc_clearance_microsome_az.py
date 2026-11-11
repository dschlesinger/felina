"""TDC AstraZeneca microsomal clearance: continuous label `Y`, 1,102 molecules."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_clearance_microsome_az/tdc_clearance_microsome_az.csv"
FILE = "metabolism/tdc_clearance_microsome_az.csv"

SPEC = DatasetSpec(
    name="tdc_clearance_microsome_az",
    group="metabolism",
    description="TDC AstraZeneca microsomal clearance, regression (1,102)",
    order=130,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_clearance_microsome_az": Source(
            kind="external",
            heads=[(r"^(Y)$", "MicrosomeClearance-AZ")],
            log10p1=True,
            description="TDC AstraZeneca microsomal clearance, log10(x+1)",
        ),
    },
)
