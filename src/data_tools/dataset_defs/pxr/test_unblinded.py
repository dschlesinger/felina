"""The test split with its labels released: the same 513 molecules, now with pEC50 and Emax.

The labels came out one phase at a time (253 molecules when phase 1 closed, 260 when phase 2 did), so the fetch
takes both files and the derive step stacks them into one table with the train split's column names. Blind
predictions can then be scored against it directly (`evaluate-models --unblinded test_unblinded`).

It declares no training source on purpose. Every molecule here is in the blinded test split, so the merge's
leakage filter would drop all of it anyway; it is for evaluation only.
"""

from pathlib import Path

from data_tools.dataset_spec import DatasetSpec
from data_tools.fetchers import csv_from

from ._reshape import REPO, stack

PHASES = ("test_phase1_raw.csv", "test_phase2_raw.csv")
URLS = (f"{REPO}/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv", f"{REPO}/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
FILE = "test_unblinded.csv"


def _fetch(data_dir: Path) -> None:
    for url, name in zip(URLS, PHASES, strict=True):
        csv_from(url, name)(data_dir)


SPEC = DatasetSpec(
    name="test_unblinded",
    group="challenge",
    description="Unblinded test split, both phases (pEC50 and Emax)",
    order=25,
    fetched=PHASES,
    fetch=_fetch,
    derived=(FILE,),
    derive=stack(PHASES, FILE),
)
