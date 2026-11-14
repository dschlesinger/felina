"""Tests for the Avalon fingerprint representations."""

import numpy as np
import polars as pl
import pytest

from data_tools.inputs import INPUT_REGISTRY
from representations.avalon import AvalonFingerprint, CountAvalonFingerprint

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


@pytest.fixture(params=[AvalonFingerprint, CountAvalonFingerprint])
def rep(request):
    return request.param()


def test_shape_and_dtype(rep):
    out = rep.transform(pl.Series(SMILES))
    assert out.shape == (3, 1024)
    assert out.dtype == np.float64


def test_feature_names_match_width(rep):
    assert len(rep.feature_names) == 1024


def test_invalid_null_and_blank_smiles_are_all_nan(rep):
    out = rep.transform(pl.Series(["CCO", "not-a-smiles", None, "   "]))
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()
    assert np.isnan(out[2]).all()
    assert np.isnan(out[3]).all()


def test_rows_follow_input_order_and_are_deterministic(rep):
    out = rep.transform(pl.Series(SMILES))
    reversed_out = rep.transform(pl.Series(SMILES[::-1]))
    assert np.array_equal(out, reversed_out[::-1])


def test_different_molecules_give_different_rows(rep):
    out = rep.transform(pl.Series(SMILES))
    assert not np.array_equal(out[0], out[1])


def test_bits_are_binary_and_counts_reach_above_one():
    bits = AvalonFingerprint().transform(pl.Series(SMILES))
    assert set(np.unique(bits)) <= {0.0, 1.0}
    counts = CountAvalonFingerprint().transform(pl.Series(SMILES))
    assert counts.max() > 1.0


def test_every_counted_bit_is_also_set_in_the_bit_vector():
    """RDKit's count fingerprint can leave a bit the bit vector sets at zero, not the reverse."""
    bits = AvalonFingerprint().transform(pl.Series(SMILES))
    counts = CountAvalonFingerprint().transform(pl.Series(SMILES))
    assert np.all(bits[counts > 0] > 0)


def test_empty_input_returns_empty_matrix(rep):
    out = rep.transform(pl.Series([], dtype=pl.Utf8))
    assert out.shape == (0, 1024)


@pytest.mark.parametrize("name", ["avalon", "count_avalon"])
def test_registered(name):
    df = pl.DataFrame({"SMILES": SMILES})
    assert name in INPUT_REGISTRY
    assert INPUT_REGISTRY[name](df).shape == (3, 1024)
