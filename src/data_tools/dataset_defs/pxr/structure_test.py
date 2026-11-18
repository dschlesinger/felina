"""The structure track's blinded ligand set: 184 ligands to pose in the PXR ligand-binding domain.

Only some rows carry a `Molecule Name`, and none carry a label, so it declares no training source. It is here
so `data fetch structure_test` puts the ligand list beside the activity data; posing it is the `src/structure/`
co-fold pipeline's job.
"""

from data_tools.dataset_spec import DatasetSpec
from data_tools.fetchers import csv_from

from ._reshape import REPO, normalise

URL = f"{REPO}/pxr-challenge_structure_TEST_BLINDED.csv"
RAW = "structure_test_raw.csv"
FILE = "structure_test.csv"

SPEC = DatasetSpec(
    name="structure_test",
    group="challenge",
    description="Structure track: blinded ligands to pose (184)",
    order=30,
    fetched=(RAW,),
    fetch=csv_from(URL, RAW),
    derived=(FILE,),
    derive=normalise(RAW, FILE),
)
