"""load_data's blind_matched strategy, the one evaluate-models exposes via --split."""

import polars as pl
import pytest

from data_tools.datasets import challenge
from data_tools.load import load_data, test_file, train_file

BENZENES = ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "COc1ccccc1", "Clc1ccccc1", "Fc1ccccc1"]
PYRIDINES = ["Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1", "COc1ccncc1", "Clc1ccncc1", "Fc1ccncc1"]
THIOPHENES = ["Cc1cccs1", "CCc1cccs1", "CCCc1cccs1", "COc1cccs1", "Clc1cccs1", "Fc1cccs1"]
ALIPHATICS = ["CCCCCCO", "CCCCCCCN", "CCCCCC(=O)O", "CCCCCCCl", "CCCCCCBr", "CCCCCCF"]


def _write(data_dir, train_smiles, blind_smiles):
    pl.DataFrame(
        {
            "SMILES": train_smiles,
            # The scored label, whatever the challenge calls it: the split only looks at scored columns.
            **{t: [5.0 + 0.1 * i for i in range(len(train_smiles))] for t in challenge().targets},
        }
    ).write_csv(data_dir / train_file())
    pl.DataFrame({"Molecule_Name": [f"B{i}" for i in range(len(blind_smiles))], "SMILES": blind_smiles}).write_csv(
        data_dir / test_file()
    )


def test_blind_matched_partitions_train_and_holds_something_out(tmp_path):
    train_smiles = BENZENES + PYRIDINES + THIOPHENES + ALIPHATICS
    _write(tmp_path, train_smiles, ["CCc1ccccc1C", "CCCCCCCO"])
    train_df, val_df = load_data(split_type="blind_matched", val_fraction=0.25, seed=0, data_dir=tmp_path)
    assert len(train_df) + len(val_df) == len(train_smiles)
    assert len(val_df) > 0
    assert not set(train_df["SMILES"]) & set(val_df["SMILES"])


def test_blind_matched_requires_the_blind_set(tmp_path):
    _write(tmp_path, BENZENES + PYRIDINES, ["Cc1ccccc1"])
    (tmp_path / test_file()).unlink()
    with pytest.raises(ValueError, match="needs the blind set as its reference"):
        load_data(split_type="blind_matched", data_dir=tmp_path)
