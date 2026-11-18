"""The PXR challenge's train split: the scored pEC50 labels.

4,140 dose-response curves against the Pregnane-X Receptor. `pEC50` is the one column the challenge scores;
the Emax columns of the same curves are a separate dataset (`emax_train`), since the scored dataset declares
the source carrying its scored heads and nothing else.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

from ._reshape import REPO, normalise

URL = f"{REPO}/pxr-challenge_TRAIN.csv"
RAW = "train_raw.csv"
FILE = "train.csv"

SPEC = DatasetSpec(
    name="train",
    group="challenge",
    description="PXR induction train split (pEC50 per compound)",
    order=10,
    fetched=(RAW,),
    fetch=csv_from(URL, RAW),
    derived=(FILE,),
    derive=normalise(RAW, FILE),
    # The scored dataset: `datasets.challenge()` resolves to this, and every evaluator reads its targets,
    # columns and blind set from here instead of holding its own copy.
    scored=True,
    targets=("pEC50",),
    label_suffix="",  # the scored column is already the name it reports under
    blind="test",
    sources={
        "challenge": Source(
            kind="inhouse",
            heads=[(r"^(pEC50)$", "{}-PXR")],
            description="scored head: PXR induction pEC50 (always included)",
        ),
    },
)
