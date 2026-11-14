"""The challenge's test split with its labels released: the same molecules as `test`, now with pIC50 per isoform.

Same columns as the direct-inhibition train split (pIC50, confidence bounds, std per isoform), so a model's test
predictions can be scored against it directly.

It declares no training source on purpose. Every molecule here is in the blinded test split, so the merge's leakage
filter would drop all of it anyway; it is for evaluation only.
"""

from data_tools.dataset_spec import DatasetSpec
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-UNBLINDED_inhibition.csv"
FILE = "test_unblinded.csv"

SPEC = DatasetSpec(
    name="test_unblinded",
    group="challenge",
    description="Unblinded test split (direct-inhibition pIC50 per isoform)",
    order=25,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
)
