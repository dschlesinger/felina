"""Atom-JEPA pretrained 3D foundation-model representations.

Fixed encoder only (no fine-tuning): embeds SMILES with the Atom-JEPA EquiformerV3 encoders into 256-dim vectors
(mean of the per-atom invariant features). The package generates the conformer itself (RDKit ETKDGv3 + MMFF94),
and embeddings depend on that conformer.

Two encoders are exposed: `molecules` (pretrained on Uni-Mol's 19M molecules) as `atom_jepa`, and `bio`
(continued on ~1.5M PCBA molecules across 1,089 bioassays) as `atom_jepa_bio`. A fine-tuned encoder
(`encoder.pt` written by `train-model --model atom_jepa[_bio]`) is selected as `atom_jepa[/path/encoder.pt]` or
`atom_jepa_bio[/path/encoder.pt]` (or `[alias]`, see foundation_alias) wherever an input name is accepted.
Reference: https://huggingface.co/atom-jepa/atom-jepa, https://github.com/khelverskovp/atom-jepa
"""

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation
from representations.foundation_alias import resolve_alias

_OUTPUT_DIM = 256
_BATCH_SIZE = 32
_SPEC = re.compile(r"^(atom_jepa_bio|atom_jepa)\[(.+)\]$")


def parse_spec(name: str) -> tuple[str, Path] | None:
    """Return (`atom_jepa` or `atom_jepa_bio`, weights path) of an `atom_jepa[/path]` / `atom_jepa[alias]` input name, else None."""
    match = _SPEC.match(name)
    return (match.group(1), Path(resolve_alias(match.group(2))).expanduser()) if match else None


def cache_name(name: str) -> str:
    """Filesystem-safe cache stem for a spec, so the cache is keyed by the weights, not just 'atom_jepa'.

    Includes size and mtime: retraining to the same path must not reuse stale embeddings.
    """
    spec = parse_spec(name)
    if spec is None:
        raise ValueError(f"not an atom_jepa[...] input: {name!r}")
    kind, path = spec
    stat = path.stat()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(path.resolve())).strip("_")[-80:]
    return f"{kind}_{slug}_{stat.st_size}_{int(stat.st_mtime)}"


class AtomJEPA(Representation):
    """256-dim embeddings from a pretrained Atom-JEPA encoder (`molecules` or `bio`).

    Invalid SMILES, and molecules the package cannot embed, produce an all-NaN row.
    """

    name: ClassVar[str] = "atom_jepa"
    variant: ClassVar[str] = "molecules"

    def __init__(self, variant: str | None = None, path: Path | None = None) -> None:
        if variant is not None:
            self.variant = variant
        self.path = path
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            import torch
            from atom_jepa import AtomJEPA as Encoder

            if self.path is not None and not self.path.exists():
                raise FileNotFoundError(f"atom_jepa weights not found: {self.path}")
            model = Encoder.from_pretrained(str(self.path) if self.path is not None else self.variant)
            model.eval()
            self._model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    def _embed(self, smiles: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self._model.embed(smiles)
        return np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 256) float64 array of Atom-JEPA embeddings."""
        self._ensure_loaded()
        smiles_list = smiles.to_list()
        out = np.full((len(smiles_list), _OUTPUT_DIM), np.nan, dtype=np.float64)

        for start in tqdm(range(0, len(smiles_list), _BATCH_SIZE), desc=f"Atom-JEPA ({self.variant}) embeddings", unit="batch"):
            chunk = smiles_list[start : start + _BATCH_SIZE]
            valid = [i for i, smi in enumerate(chunk) if isinstance(smi, str) and Chem.MolFromSmiles(smi) is not None]
            if not valid:
                continue
            try:
                embeddings = self._embed([chunk[i] for i in valid])
                rows = list(zip(valid, embeddings))
            except (ValueError, RuntimeError):
                # One molecule (e.g. a failed conformer) sank the batch: embed individually, NaN the failures.
                rows = []
                for i in valid:
                    try:
                        rows.append((i, self._embed([chunk[i]])[0]))
                    except (ValueError, RuntimeError):
                        pass
            for i, embedding in rows:
                if embedding.shape != (_OUTPUT_DIM,):
                    raise ValueError(f"Unexpected Atom-JEPA embedding shape: {embedding.shape}")
                out[start + i] = embedding

        return out


class AtomJEPABio(AtomJEPA):
    """Atom-JEPA-Bio: the `molecules` encoder continued on PCBA bioassays."""

    name: ClassVar[str] = "atom_jepa_bio"
    variant: ClassVar[str] = "bio"
