"""Dimensionality reduction between a representation and the model, chosen with `--dimred`.

    pca[N]   exact PCA via torch SVD (GPU when available), fit on the training rows' features
    umap[N]  UMAP (umap-learn, CPU), fit on the training rows' features
    pls[N]   PLS x-scores, fit on the training rows' features *and labels*

Unsupervised reducers (pca, umap) are fit once per split and input on every training row. Supervised
ones (pls) see labels, so they are fit per target and, under cross-validation, per fold on that fold's
training rows only; otherwise the held-out fold's labels would leak into its own features.

Fitted reducers are cached by content: a digest of the matrix they were fit on (and the labels, for
pls) plus the spec. A different split, seed, input or retrained weights therefore gets its own entry,
and re-running the same evaluation loads instead of refitting. UMAP's outputs are cached too, since
its transform is slow; the others are cheap matrix products.
"""

import hashlib
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

UNSUPERVISED = ("pca", "umap")
SUPERVISED = ("pls",)
METHODS = (*UNSUPERVISED, *SUPERVISED)
# Bump a method's version whenever _build changes what it fits, so older cached fits stop matching.
_VERSIONS = {"pca": 1, "umap": 1, "pls": 2}
# Columns with less training variance than this are dropped before PLS.
_MIN_VARIANCE = 1e-8
# Methods whose transform is costly enough to be worth caching on disk.
_CACHE_OUTPUTS = {"umap"}

_SPEC = re.compile(rf"^({'|'.join(METHODS)})\[(\d+)\]$")


@dataclass(frozen=True)
class DimRed:
    """A parsed `--dimred` spec such as `pca[200]`."""

    method: str
    n_components: int

    def __str__(self) -> str:
        return f"{self.method}[{self.n_components}]"

    @property
    def supervised(self) -> bool:
        return self.method in SUPERVISED


def parse_spec(text: str) -> DimRed:
    """Parse `method[N]`; raises ValueError naming the accepted forms."""
    match = _SPEC.match(text.strip())
    if match is None or int(match.group(2)) < 1:
        raise ValueError(f"Bad --dimred {text!r}: expected one of {', '.join(f'{m}[N]' for m in METHODS)} with N >= 1")
    return DimRed(match.group(1), int(match.group(2)))


class TorchPCA(TransformerMixin, BaseEstimator):
    """Exact PCA via torch SVD; runs on the GPU when one is available, else CPU.

    The fitted mean and components are kept as numpy arrays, so a pickled fit loads on a machine
    without a GPU. n_components is clamped to the data shape.
    """

    def __init__(self, n_components: int = 256, device: str = "auto") -> None:
        self.n_components = n_components
        self.device = device

    def _device(self):
        import torch

        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y=None) -> "TorchPCA":
        import torch

        device = self._device()
        try:
            mean, vt = self._svd(X, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("TorchPCA: CUDA out of memory; fitting on CPU", file=sys.stderr)
            mean, vt = self._svd(X, torch.device("cpu"))
        self.mean_ = mean
        self.components_ = vt[: min(self.n_components, *np.shape(X))]
        return self

    @staticmethod
    def _svd(X: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
        import torch

        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        mean = x.mean(dim=0)
        _, _, vt = torch.linalg.svd(x - mean, full_matrices=False)
        return mean.cpu().numpy(), vt.cpu().numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) - self.mean_) @ self.components_.T


def _build(spec: DimRed, shape: tuple[int, int]):
    n_rows, n_features = shape
    n = spec.n_components
    if spec.method == "pca":
        k = min(n, n_rows, n_features)
        model = TorchPCA(n_components=k)
    elif spec.method == "umap":
        import umap

        k = min(n, n_rows - 2)
        model = umap.UMAP(n_components=k, n_jobs=-1)
    else:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.pipeline import Pipeline

        k = min(n, n_rows, n_features)
        # PLS scales every column to unit variance, so a column nearly constant on the training rows
        # becomes a huge multiplier for any new molecule that differs there; drop those first.
        model = Pipeline([
            ("drop_constant", VarianceThreshold(_MIN_VARIANCE)),
            ("pls", PLSRegression(n_components=k, scale=True)),
        ])
    if k != n:
        print(f"--dimred {spec}: clamped to {k} components for a {n_rows}x{n_features} matrix", file=sys.stderr)
    return model


def _digest(*arrays: np.ndarray, extra: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(extra.encode())
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(f"|{array.dtype}{array.shape}|".encode())
        h.update(array.data)
    return h.hexdigest()


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        write(stream)
    tmp.replace(path)


class Reducer:
    """A fitted reducer whose transform output is cached when the method is slow."""

    def __init__(self, spec: DimRed, model, key: str, cache_dir: Path) -> None:
        self.spec = spec
        self.model = model
        self.key = key
        self.cache_dir = cache_dir

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.spec.method not in _CACHE_OUTPUTS:
            return np.asarray(self.model.transform(X), dtype=np.float64)
        path = self.cache_dir / f"{self.spec.method}_{self.key}_{_digest(X, extra=self.key)}.npy"
        if path.exists():
            return np.load(path)
        out = np.asarray(self.model.transform(X), dtype=np.float64)
        _atomic_write(path, lambda stream: np.save(stream, out))
        return out


def fit(spec: DimRed, X: np.ndarray, y: np.ndarray | None = None, *, cache_dir: Path) -> Reducer:
    """Fit (or load the cached fit of) `spec` on X, and on y for supervised methods."""
    if spec.supervised and y is None:
        raise ValueError(f"--dimred {spec} is supervised and needs the training labels")
    X = np.asarray(X, dtype=np.float64)
    arrays = (X, np.asarray(y, dtype=np.float64)) if spec.supervised else (X,)
    key = _digest(*arrays, extra=f"{spec}|v{_VERSIONS[spec.method]}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{spec.method}_{key}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            model = pickle.load(stream)
    else:
        model = _build(spec, X.shape)
        if spec.supervised:
            model.fit(X, y)
        else:
            model.fit(X)
        _atomic_write(path, lambda stream: pickle.dump(model, stream))
    return Reducer(spec, model, key, cache_dir)
