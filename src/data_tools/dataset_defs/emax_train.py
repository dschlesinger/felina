"""Emax train split: how far inhibition goes at saturation, under both conditions."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_Emax.csv"
FILE = "emax_train.csv"

SPEC = DatasetSpec(
    name="emax_train",
    group="challenge",
    description="Emax train split",
    order=40,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "emax": Source(
            kind="inhouse",
            heads=[
                (r"^(.+)_EmaxVsPosCtrl_TDI_condition$", "{}-Emax-TDI"),
                (r"^(.+)_EmaxVsPosCtrl_direct_inhibition$", "{}-Emax-Direct"),
            ],
            description="Emax vs positive control, TDI and direct conditions",
        ),
    },
)
