"""Uni-Mol molecular embeddings.

Uses a Uni-Mol v2 encoder (MODEL_SIZE) to generate molecular representations from SMILES. By default this is
the pretrained encoder, selected as `unimol`; its rows are cached as PRETRAINED_CACHE_STEM so they never mix with
the 512-dimensional rows cached from the earlier v1 encoder. A fine-tuned model is selected as `unimol[/path/to/model_0.pth]`
(or `unimol[alias]`, see foundation_alias) wherever an input name is accepted: the weights are the `model_0.pth`
written by `fm train --model unimol`, and the fingerprint is the encoder's CLS embedding; the regression head is not used.
"""

import re
from pathlib import Path
from typing import ClassVar
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation
from representations.foundation_alias import resolve_alias


MODEL_SIZE = "310m"  # a unimol_tools v2 size: 84m, 164m, 310m, 570m or 1.1B
_EMBED_DIMS = {"84m": 768, "164m": 768, "310m": 1024, "570m": 1536, "1.1B": 1536}
_OUTPUT_DIM = _EMBED_DIMS[MODEL_SIZE]
PRETRAINED_CACHE_STEM = f"unimolv2_{MODEL_SIZE}"
_BATCH_SIZE = 32
# v2 attends over atom pairs, so its activations grow with the square of the largest molecule in a batch:
# the whole conformer chunk at once runs a 4 GB GPU out of memory, and the encoder gets it in smaller pieces.
# The bigger checkpoints hold more of the card themselves, so they take the chunk in smaller pieces still.
_INFER_BATCH_SIZES = {"84m": 8, "164m": 8, "310m": 4, "570m": 2, "1.1B": 1}
_INFER_BATCH_SIZE = _INFER_BATCH_SIZES[MODEL_SIZE]
_SPEC = re.compile(r"^unimol\[(.+)\]$")


def parse_spec(name: str) -> Path | None:
    """Return the weights path of a `unimol[/path]` or `unimol[alias]` input name, or None if it isn't one."""
    match = _SPEC.match(name)
    return Path(resolve_alias(match.group(1))).expanduser() if match else None


def cache_prefix(name: str) -> str:
    """The cache stem without the weights' size and mtime, derived from the path alone (no filesystem access)."""
    path = parse_spec(name)
    if path is None:
        raise ValueError(f"not a unimol[...] input: {name!r}")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(path.resolve())).strip("_")[-80:]
    return f"{PRETRAINED_CACHE_STEM}_{slug}"


def cache_name(name: str) -> str:
    """Filesystem-safe cache stem for a spec, so the cache is keyed by the weights, not just 'unimol'.

    Includes size and mtime: retraining to the same path must not reuse stale embeddings.
    """
    stat = parse_spec(name).stat()
    return f"{cache_prefix(name)}_{stat.st_size}_{int(stat.st_mtime)}"


def _encoder_only(path: Path) -> Path:
    """Copy of a fine-tuned checkpoint without its regression head, which UniMolRepr's 1-output head can't load.

    Fine-tuning on several targets leaves an (n_targets, dim) output layer; the embeddings come from the encoder
    alone. The v2 loader is non-strict, so the dropped head keys just keep their initial values.
    """
    import tempfile

    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    # The v2 loader is non-strict, so v1 weights would load as nothing at all and embed noise in silence.
    if not any(k.startswith("se3_invariant_kernel.") for k in state):
        raise ValueError(
            f"{path} is not a Uni-Mol v2 checkpoint (no se3_invariant_kernel weights). It was most likely "
            "fine-tuned by an earlier `fm train --model unimol`, which trained Uni-Mol v1; retrain it."
        )
    state = {k: v for k, v in state.items() if not k.startswith("classification_head.")}
    stripped = Path(tempfile.mkdtemp()) / path.name
    torch.save({"model_state_dict": state} if "model_state_dict" in checkpoint else state, stripped)
    return stripped


class UniMol(Representation):
    """Generate Uni-Mol v2 embeddings: pretrained, or the fine-tuned model at `path`."""

    name: ClassVar[str] = "unimol"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._model = None 

    def _ensure_loaded(self) -> None:
        """Load the pretrained Uni-Mol model only when needed."""
        if self._model is None:
            import torch
            from unimol_tools import UniMolRepr

            weights = {}
            if self.path is not None:
                if not self.path.exists():
                    raise FileNotFoundError(f"unimol weights not found: {self.path}")
                weights = {"pretrained_model_path": str(_encoder_only(self.path))}
            self._model = UniMolRepr(
                data_type="molecule",
                model_name="unimolv2",
                model_size=MODEL_SIZE,
                batch_size=_INFER_BATCH_SIZE,
                use_cuda=torch.cuda.is_available(),
                **weights,
            )

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return deterministic Uni-Mol v2 embeddings, one _OUTPUT_DIM row per SMILES (NaN where embedding failed)."""
        self._ensure_loaded()

        smiles_list = smiles.to_list()
        out = np.full(
            (len(smiles_list), _OUTPUT_DIM),
            np.nan,
            dtype=np.float64,
        )

        failed_conformers = 0

        for start in tqdm(
            range(0, len(smiles_list), _BATCH_SIZE),
            desc="Uni-Mol embeddings",
            unit="batch",
        ):
            chunk = smiles_list[start:start + _BATCH_SIZE]

            valid_idx = []
            mols = []

            for i, smi in enumerate(chunk):
                if not isinstance(smi, str):
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                try:
                    mol = Chem.AddHs(mol)

                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42

                    if AllChem.EmbedMolecule(mol, params) != 0:
                        failed_conformers += 1
                        continue

                    if not AllChem.MMFFHasAllMoleculeParams(mol):
                        failed_conformers += 1
                        continue

                    if AllChem.MMFFOptimizeMolecule(
                        mol, maxIters=1000
                    ) != 0:
                        failed_conformers += 1
                        continue

                    if not np.isfinite(mol.GetConformer().GetPositions()).all():
                        failed_conformers += 1
                        continue

                    valid_idx.append(i)
                    mols.append(mol)

                except (ValueError, RuntimeError, TypeError):
                    failed_conformers += 1
                    continue

            if not valid_idx:
                continue

            # The molecules go in whole, not as (atoms, coordinates): Uni-Mol v2 features the bond graph as well
            # as the geometry, and unimol_tools rebuilds a bondless molecule from bare atoms and coordinates.
            embeddings = np.asarray(
                self._model.get_repr(pd.DataFrame({"ROMol": mols})),
                dtype=np.float64,
            )

            if embeddings.shape != (len(valid_idx), _OUTPUT_DIM):
                raise ValueError(
                    f"Unexpected Uni-Mol embedding shape: {embeddings.shape}"
                )

            for row, i in enumerate(valid_idx):
                if np.isfinite(embeddings[row]).all():
                    out[start + i] = embeddings[row]

        print(f"Failed 3D conformers: {failed_conformers}/{len(smiles_list)}")

        return out