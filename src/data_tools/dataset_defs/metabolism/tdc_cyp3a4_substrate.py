"""TDC CYP3A4 substrate (Carbon-Mangels): binary label `Y`, 670 molecules.

Substrate, not inhibitor: whether the isoform metabolizes the molecule, not whether the molecule blocks it.
A separate head, never pooled with the scored inhibition columns.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_cyp3a4_substrate_carbonmangels/tdc_cyp3a4_substrate_carbonmangels.csv"
FILE = "metabolism/tdc_cyp3a4_substrate.csv"

SPEC = DatasetSpec(
    name="tdc_cyp3a4_substrate",
    group="metabolism",
    description="TDC CYP3A4 substrate, binary (670)",
    order=170,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_cyp3a4_substrate": Source(
            kind="external",
            heads=[(r"^(Y)$", "CYP3A4-Substrate")],
            task="classification",
            description="TDC CYP3A4 substrate, binary",
        ),
    },
)
