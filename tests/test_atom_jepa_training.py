import numpy as np
import pandas as pd
import pytest
import torch

from models.foundation_model import KINDS
from models.training import BACKENDS
from models.training.atom_jepa import masked_abs_error, split_holdout
from representations.atom_jepa import parse_spec


def test_backends_registered():
    assert BACKENDS["atom_jepa"].variant == "molecules"
    assert BACKENDS["atom_jepa_bio"].variant == "bio"


def test_kinds_registered():
    for name in ("atom_jepa", "atom_jepa_bio"):
        kind = KINDS[name]
        run = kind.run_pattern.fullmatch("lr_0.0001_seed_3")
        assert run is not None and kind.weights_file.name == "encoder.pt"


def test_parse_spec(tmp_path):
    assert parse_spec("atom_jepa") is None
    assert parse_spec("atom_jepa[/x/encoder.pt]")[0] == "atom_jepa"
    assert parse_spec("atom_jepa_bio[/x/encoder.pt]")[0] == "atom_jepa_bio"


def test_masked_abs_error_ignores_nan():
    pred = torch.tensor([[1.0, 5.0], [2.0, 5.0]])
    y = torch.tensor([[0.0, float("nan")], [4.0, float("nan")]])
    err, count = masked_abs_error(pred, y)
    assert err.tolist() == [3.0, 0.0] and count.tolist() == [2, 0]


def test_split_holdout_keeps_scaffolds_and_blocks_together():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "C1CCCCC1C", "C1CCCCC1CC", "CCO", "CCN", "CCC", "CCCC", "c1ccncc1C", "c1ccncc1CC"]
    df = pd.DataFrame({"SMILES": smiles, "inchikey_block": [f"b{i}" for i in range(len(smiles))]})
    mask = split_holdout(df, 0.3, seed=0)
    assert mask.any() and not mask.all()
    assert mask.tolist() == split_holdout(df, 0.3, seed=0).tolist()  # deterministic
    for a, b in [(0, 1), (2, 3), (8, 9)]:  # same scaffold, same side
        assert mask[a] == mask[b]


def test_split_holdout_rejects_bad_fraction():
    df = pd.DataFrame({"SMILES": ["CCO"], "inchikey_block": ["b"]})
    with pytest.raises(ValueError):
        split_holdout(df, 1.0, seed=0)
