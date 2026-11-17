"""RDKit Avalon fingerprint representation."""

from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from tqdm import tqdm

from representations.base import Representation

_N_BITS = 1024


class AvalonFingerprint(Representation):
    """1024-bit Avalon fingerprint.

    Avalon enumerates its own feature classes (atom pairs and triplets, ring and
    bond paths, substructure fragments) in a fixed hand-written set rather than
    hashing circular environments, so it is complementary to Morgan.

    Invalid, null, and blank SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "avalon"

    @property
    def feature_names(self) -> list[str]:
        """Return names in Avalon bit order."""
        return [f"Avalon_{i}" for i in range(_N_BITS)]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 1024) float64 matrix in input row order."""
        out = np.full((len(smiles), _N_BITS), np.nan, dtype=np.float64)
        for i, smi in enumerate(
            tqdm(smiles.to_list(), desc="Avalon fingerprint", unit="mol", leave=True)
        ):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=_N_BITS)
                out[i] = np.asarray(fp, dtype=np.float64)
        return out


class CountAvalonFingerprint(Representation):
    """1024-dimensional count Avalon fingerprint.

    Same feature set as :class:`AvalonFingerprint`, with per-bit occurrence
    counts instead of presence bits.

    Invalid, null, and blank SMILES produce an all-NaN row.
    """

    name: ClassVar[str] = "count_avalon"

    @property
    def feature_names(self) -> list[str]:
        """Return names in Avalon bit order."""
        return [f"CountAvalon_{i}" for i in range(_N_BITS)]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 1024) float64 matrix in input row order."""
        out = np.full((len(smiles), _N_BITS), np.nan, dtype=np.float64)
        for i, smi in enumerate(
            tqdm(smiles.to_list(), desc="Count-Avalon fingerprint", unit="mol", leave=True)
        ):
            if smi is None or not smi.strip():
                continue
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                continue
            fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=_N_BITS)
            row = np.zeros(_N_BITS, dtype=np.float64)
            for bit, count in fp.GetNonzeroElements().items():
                row[bit] = count
            out[i] = row
        return out
