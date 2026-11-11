"""Time-dependent-inhibition train split: the same molecules, measured after preincubation."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv"
FILE = "tdi_train.csv"

SPEC = DatasetSpec(
    name="tdi_train",
    group="challenge",
    description="Time-dependent inhibition (TDI) train split",
    order=30,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "tdi": Source(
            kind="inhouse",
            heads=[(r"^(.+)_pIC50_TDI_condition$", "{}-TDI")],
            description="pIC50 under time-dependent-inhibition conditions",
        ),
    },
)
