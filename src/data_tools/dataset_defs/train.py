"""The challenge's direct-inhibition train split: the scored labels."""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv"
FILE = "train.csv"
SUFFIX = "_pIC50_direct_inhibition"

SPEC = DatasetSpec(
    name="train",
    group="challenge",
    description="Direct-inhibition train split (pIC50 per isoform)",
    order=10,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    # The scored dataset: `datasets.challenge()` resolves to this, and every evaluator reads its targets,
    # columns and blind set from here instead of holding its own copy.
    scored=True,
    targets=tuple(f"CYP{iso}{SUFFIX}" for iso in ("1A2", "2C9", "2D6", "3A4")),
    label_suffix=SUFFIX,
    blind="test",
    sources={
        "challenge": Source(
            kind="inhouse",
            heads=[(r"^(.+)_pIC50_direct_inhibition$", "{}-Direct")],
            description="scored heads: direct-inhibition pIC50 (always included)",
        ),
    },
)
