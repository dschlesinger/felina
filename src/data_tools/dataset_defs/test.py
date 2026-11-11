"""The challenge's blinded test split: molecules only, no labels.

It declares no training source. It is the prediction target, and it is also the reference the merge uses to drop
training molecules whose skeleton appears here.
"""

from data_tools.dataset_spec import DatasetSpec
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv"
FILE = "test.csv"

SPEC = DatasetSpec(
    name="test",
    group="challenge",
    description="Blinded test split",
    order=20,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
)
