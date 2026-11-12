"""Training backends behind the `train-model` CLI.

To add a model: create a module here with a Backend subclass (see base.py), then
register it in BACKENDS below. Shared data handling lives in common.py.
"""

from models.training.atom_jepa import AtomJEPABackend
from models.training.base import Backend
from models.training.chemprop import ChempropBackend
from models.training.unimol import UnimolBackend

BACKENDS: dict[str, Backend] = {
    "chemprop": ChempropBackend("chemprop"),
    "chemeleon": ChempropBackend("chemeleon", foundation="chemeleon"),  # chemprop started from CheMeleon weights
    "unimol": UnimolBackend("unimol"),
    "atom_jepa": AtomJEPABackend("atom_jepa", "molecules"),
    "atom_jepa_bio": AtomJEPABackend("atom_jepa_bio", "bio"),  # same, started from the PCBA bioassay-trained encoder
}
