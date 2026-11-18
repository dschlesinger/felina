"""The PXR challenge's blinded test split: 513 molecules, no labels.

It declares no training source. It is the prediction target, and it is also the reference the merge uses to drop
training molecules whose skeleton appears here.
"""

from data_tools.dataset_spec import DatasetSpec
from data_tools.fetchers import csv_from

from ._reshape import REPO, normalise

URL = f"{REPO}/pxr-challenge_TEST_BLINDED.csv"
RAW = "test_raw.csv"
FILE = "test.csv"

SPEC = DatasetSpec(
    name="test",
    group="challenge",
    description="Blinded test split (513 molecules)",
    order=20,
    fetched=(RAW,),
    fetch=csv_from(URL, RAW),
    derived=(FILE,),
    derive=normalise(RAW, FILE),
)
