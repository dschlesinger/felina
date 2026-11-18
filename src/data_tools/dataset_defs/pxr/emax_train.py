"""Emax from the train split: how far induction goes at saturation, as an auxiliary head.

The same curves the scored pEC50 comes from, read at their ceiling rather than their midpoint, in two
parameterisations: a log2 fold change over baseline, and the same ceiling against the positive control
(rifampicin = 1). A model that knows how hard a compound can drive PXR has learned something about where its
curve sits, so these train alongside the scored head (`--datasets challenge emax`).

It derives its own file from its own copy of the train CSV rather than reading the scored dataset's table: a
dataset owns the files it writes, and `data remove emax_train` must not delete the scored labels.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

from ._reshape import REPO, normalise

URL = f"{REPO}/pxr-challenge_TRAIN.csv"
RAW = "emax_train_raw.csv"
FILE = "emax_train.csv"
KEEP = (
    "Molecule_Name",
    "SMILES",
    "Emax_log2FC",
    "Emax_vs_pos_ctrl",
    "Emax_log2FC_std_error",
    "Emax_vs_pos_ctrl_std_error",
)

SPEC = DatasetSpec(
    name="emax_train",
    group="challenge",
    description="Emax of the train split (induction ceiling, two parameterisations)",
    order=15,
    fetched=(RAW,),
    fetch=csv_from(URL, RAW),
    derived=(FILE,),
    derive=normalise(RAW, FILE, keep=KEEP),
    sources={
        "emax": Source(
            kind="inhouse",
            heads=[(r"^(Emax_log2FC)$", "{}-PXR"), (r"^(Emax_vs_pos_ctrl)$", "{}-PXR")],
            description="auxiliary heads: induction ceiling, as log2 fold change and vs the positive control",
        ),
    },
)
