"""Monroe pretrained foundation-model representation.

Fixed encoder only (no fine-tuning): downloads the released Monroe checkpoint and uses it to
fingerprint SMILES into 720-dim graph embeddings. Monroe is a GRIT graph transformer pretrained
on PM6 quantum-chemical properties plus PCBA bioassays with a conformer-denoising objective.
Reference: https://github.com/blazejba/Monroe (NeurIPS 2026, arXiv:2608.18982)

Selected as `monroe`. A different checkpoint directory (config.json + weights.pt) is selected as
`monroe[/path/to/checkpoint]` (or `monroe[alias]`, see foundation_alias) wherever an input name is accepted.

Monroe reads 3D structure, so a molecule whose conformer generation fails -- RDKit falls back to a
flat 2D layout -- gets an all-NaN row rather than an unreliable embedding, as does one the
featurizer cannot handle at all.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import polars as pl
from tqdm import tqdm

from representations.base import Representation
from representations.foundation_alias import resolve_alias

# Pinned to the commit the weights were released at; the LFS blob is served from media.githubusercontent.com
# (raw.githubusercontent.com returns the LFS pointer text, not the file).
_COMMIT = "de075eabe98e4adbbedbc86ca9e877f4bfd9b629"
_WEIGHTS_URL = f"https://media.githubusercontent.com/media/blazejba/Monroe/{_COMMIT}/checkpoint/weights.pt"
_CONFIG_URL = f"https://raw.githubusercontent.com/blazejba/Monroe/{_COMMIT}/checkpoint/config.json"
# oid from the checkpoint's Git LFS pointer at _COMMIT
_WEIGHTS_SHA256 = "df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a"
_WEIGHTS_BYTES = 300057823

# Relative to the working directory, like data/features, so it lands on project storage rather than a
# quota-limited home. Override with MONROE_HOME.
_CHECKPOINT_DIR = Path(os.environ.get("MONROE_HOME") or ".monroe")

_OUTPUT_DIM = 720
_BATCH_SIZE = 32
_CHUNK = 512  # molecules featurized per process-pool round
_SPEC = re.compile(r"^monroe\[(.+)\]$")


def parse_spec(name: str) -> Path | None:
    """Return the checkpoint directory of a `monroe[/path]` or `monroe[alias]` input name, or None if it isn't one."""
    match = _SPEC.match(name)
    return Path(resolve_alias(match.group(1))).expanduser() if match else None


def cache_prefix(name: str) -> str:
    """The cache stem without the checkpoint's size and mtime, derived from the path alone (no filesystem access)."""
    path = parse_spec(name)
    if path is None:
        raise ValueError(f"not a monroe[...] input: {name!r}")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(path.resolve())).strip("_")[-80:]
    return f"monroe_{slug}"


def cache_name(name: str) -> str:
    """Filesystem-safe cache stem for a spec, so the cache is keyed by the weights, not just 'monroe'.

    Includes size and mtime: retraining to the same path must not reuse stale embeddings.
    """
    path = parse_spec(name)
    weights = path / "weights.pt" if path.is_dir() else path
    stat = weights.stat()
    return f"{cache_prefix(name)}_{stat.st_size}_{int(stat.st_mtime)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, desc: str) -> None:
    """Download to a temporary name beside dest, then rename, so an interrupted run leaves no half file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tqdm(desc=desc, unit="B", unit_scale=True, unit_divisor=1024) as bar:

        def hook(blocks: int, block_size: int, total: int) -> None:
            if total > 0:
                bar.total = total
            bar.update(blocks * block_size - bar.n)

        urlretrieve(url, tmp, reporthook=hook)
    tmp.rename(dest)


def ensure_checkpoint(directory: Path = _CHECKPOINT_DIR) -> Path:
    """Download the released Monroe checkpoint if it isn't there yet, and return its directory.

    A weights file of the wrong size or digest is re-downloaded, so a truncated or corrupted
    earlier attempt repairs itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    weights_path = directory / "weights.pt"

    if not config_path.is_file():
        _fetch(_CONFIG_URL, config_path, "Monroe config")

    if weights_path.is_file():
        if weights_path.stat().st_size == _WEIGHTS_BYTES and _sha256(weights_path) == _WEIGHTS_SHA256:
            return directory
        print(f"Monroe weights at {weights_path} are incomplete or corrupted; downloading again")
        weights_path.unlink()

    _fetch(_WEIGHTS_URL, weights_path, "Monroe weights (286 MB)")
    if _sha256(weights_path) != _WEIGHTS_SHA256:
        weights_path.unlink()
        raise RuntimeError(f"downloaded Monroe weights do not match the expected sha256 ({_WEIGHTS_URL})")
    return directory


def _is_flat(pos: np.ndarray) -> bool:
    """True if every atom sits at z=0, i.e. RDKit fell back to a 2D layout rather than a conformer."""
    return pos.ndim == 2 and pos.shape[0] > 0 and bool(np.all(pos[:, 2] == 0))


class Monroe(Representation):
    """720-dim graph embeddings from the pretrained Monroe GRIT encoder.

    The released checkpoint is downloaded on first use (override the location with $MONROE_HOME);
    pass `path` for a different checkpoint directory. Molecules that fail featurization, or that
    only get a flat 2D layout, produce an all-NaN row.
    """

    name: ClassVar[str] = "monroe"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._encoder = None

    def _ensure_loaded(self) -> None:
        if self._encoder is None:
            import torch
            from monroe.model.ckpt import load_ckpt

            if self.path is None:
                directory = ensure_checkpoint()
            else:
                directory = self.path
                if not (directory / "config.json").is_file():
                    raise FileNotFoundError(f"monroe checkpoint not found: {directory}/config.json")
            encoder = load_ckpt(str(directory))
            encoder.eval()
            self._encoder = encoder.to("cuda" if torch.cuda.is_available() else "cpu")

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, 720) float64 array of Monroe graph embeddings."""
        import torch
        from torch_geometric.data import Batch

        from monroe.eval.embed import featurize_smiles, to_pyg

        self._ensure_loaded()
        device = next(self._encoder.parameters()).device

        smiles_list = smiles.to_list()
        out = np.full((len(smiles_list), _OUTPUT_DIM), np.nan, dtype=np.float64)
        # One embedding per unique SMILES: featurization (conformer generation) dominates the cost.
        unique = list(dict.fromkeys(s for s in smiles_list if isinstance(s, str)))
        vectors: dict[str, np.ndarray] = {}
        failed = 0
        flat = 0

        for start in tqdm(range(0, len(unique), _CHUNK), desc="Monroe embeddings", unit="chunk"):
            chunk = unique[start : start + _CHUNK]
            built = featurize_smiles(chunk)
            failed += len(chunk) - len(built)

            keep = []
            for smi, graph in built:
                if _is_flat(np.asarray(graph["pos_rdkit"], dtype=np.float32)):
                    # Monroe is structure-aware; an embedding from a planar layout is unreliable.
                    flat += 1
                    continue
                keep.append((smi, graph))
            if not keep:
                continue

            graphs = [to_pyg(graph) for _, graph in keep]
            for begin in range(0, len(graphs), _BATCH_SIZE):
                batch = Batch.from_data_list(graphs[begin : begin + _BATCH_SIZE]).to(device)
                with torch.no_grad():
                    graph_embedding, _ = self._encoder(batch)
                embeddings = graph_embedding.cpu().numpy()
                if embeddings.shape[1] != _OUTPUT_DIM:
                    raise ValueError(f"Unexpected Monroe embedding shape: {embeddings.shape}")
                for row, (smi, _) in enumerate(keep[begin : begin + _BATCH_SIZE]):
                    if np.isfinite(embeddings[row]).all():
                        vectors[smi] = embeddings[row]

        for i, smi in enumerate(smiles_list):
            vector = vectors.get(smi) if isinstance(smi, str) else None
            if vector is not None:
                out[i] = vector

        if failed or flat:
            print(f"Monroe: {failed} unfeaturizable, {flat} 2D-fallback (NaN) of {len(unique)} unique molecules")
        return out
