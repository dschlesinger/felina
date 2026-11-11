"""TDC CYP2C9 substrate (Carbon-Mangels): binary label `Y`, about 670 molecules.

Substrate, not inhibitor: whether the isoform metabolizes the molecule, not whether the molecule blocks it.
A separate head, never pooled with the scored inhibition columns.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/scikit-fingerprints/TDC_cyp2c9_substrate_carbonmangels/tdc_cyp2c9_substrate_carbonmangels.csv"
FILE = "metabolism/tdc_cyp2c9_substrate.csv"

SPEC = DatasetSpec(
    name="tdc_cyp2c9_substrate",
    group="metabolism",
    description="TDC CYP2C9 substrate, binary (about 670)",
    order=150,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdc_cyp2c9_substrate": Source(
            kind="external",
            heads=[(r"^(Y)$", "CYP2C9-Substrate")],
            task="classification",
            description="TDC CYP2C9 substrate, binary",
        ),
    },
)
