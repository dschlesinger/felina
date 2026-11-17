"""Structure standardization applied to SMILES *before* featurization.

This is a different job from `data_tools.standardize`, which builds InChIKey
identity keys for cross-source deduplication and leakage control. Those keys
decide which rows are the same molecule; the modes here decide what string a
featurizer actually sees, and therefore what the feature cache is keyed on.

Without this, the same molecule spelled two ways (salt vs free base in train
and blind, explicit vs implicit charge between sources) produces two different
cache rows and two different feature vectors. Running every dataset through one
mode makes the spelling consistent everywhere features are computed.

Modes (`--standardization`):

  none      pass the SMILES through untouched.
  normal    largest fragment, then RDKit canonical SMILES. The default: this is
            what the project has always done for its identity keys, so it is
            the behaviour every recorded result was produced under.
  advanced  rdMolStandardize Cleanup -> FragmentParent -> Uncharger, then
            canonical SMILES. Normalizes functional groups, strips counter-ions
            by molecular weight rather than string length, and neutralizes
            charges. Competitor reports (Preservea, OstarkovSN) used this chain.

Adding a mode is one entry in `_MODES`; nothing else in the project needs to
know the list.

The mode is process-global (`set_mode`) and is mirrored into the environment so
subprocess backends inherit it. The feature cache is content-addressed by the
SMILES string, so two modes never corrupt each other's rows: a mode that emits a
different string simply gets a different cache entry.
"""

import os
from typing import Callable, Iterable, Optional

import polars as pl

from data_tools.standardize import _largest_fragment

ENV_VAR = "CYP_STANDARDIZATION"
DEFAULT_MODE = "normal"


def _none(smiles: str) -> Optional[str]:
    """Pass the SMILES through exactly as written."""
    return smiles


def _normal(smiles: str) -> Optional[str]:
    """Largest fragment, then RDKit canonical SMILES: the project's long-standing behaviour.

    Salt stripping picks the longest *string* component, which is what
    `data_tools.standardize` has always done; `advanced` is the mode that picks
    by molecular weight instead.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - rdkit is a hard dep at runtime
        raise ImportError("rdkit is required for standardization") from exc

    mol = Chem.MolFromSmiles(_largest_fragment(smiles.strip()))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _advanced(smiles: str) -> Optional[str]:
    """Cleanup -> FragmentParent -> Uncharger, then canonical SMILES.

    Cleanup normalizes functional-group representations (nitro, azide, ...),
    FragmentParent takes the parent by molecular weight so a heavy counter-ion
    cannot win, and Uncharger neutralizes what can be neutralized.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
        from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - rdkit is a hard dep at runtime
        raise ImportError("rdkit is required for standardization") from exc

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:  # pylint: disable=broad-except  # a molecule RDKit cannot standardize
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


_MODES: dict[str, Callable[[str], Optional[str]]] = {
    "none": _none,
    "normal": _normal,
    "advanced": _advanced,
}

MODES = tuple(_MODES)

_mode: str = DEFAULT_MODE
_memo: dict[tuple[str, str], str] = {}


def resolve(mode: Optional[str] = None) -> str:
    """Return the mode to use: the argument, else the process mode, validated."""
    chosen = _mode if mode is None else mode
    if chosen not in _MODES:
        raise ValueError(f"Unknown standardization mode {chosen!r}; choose from {', '.join(MODES)}")
    return chosen


def set_mode(mode: str) -> str:
    """Set the process-wide mode and mirror it to the environment for subprocesses."""
    global _mode  # pylint: disable=global-statement
    if mode not in _MODES:
        raise ValueError(f"Unknown standardization mode {mode!r}; choose from {', '.join(MODES)}")
    _mode = mode
    os.environ[ENV_VAR] = mode
    return mode


def get_mode() -> str:
    """Return the process-wide mode."""
    return _mode


def standardize_one(smiles: str, mode: Optional[str] = None) -> str:
    """Return the standardized SMILES, or the input unchanged when it cannot be standardized.

    Unparsable input is returned as given rather than dropped, so a molecule that
    RDKit rejects reaches the featurizer exactly as it would with `none` and fails
    (or is imputed) there, as it did before this module existed.
    """
    chosen = resolve(mode)
    if chosen == "none" or not smiles or not smiles.strip():
        return smiles
    key = (chosen, smiles)
    hit = _memo.get(key)
    if hit is not None:
        return hit
    result = _MODES[chosen](smiles) or smiles
    _memo[key] = result
    return result


def standardize_many(smiles_iter: Iterable[str], mode: Optional[str] = None) -> list[str]:
    """Vectorized `standardize_one`."""
    chosen = resolve(mode)
    if chosen == "none":
        return list(smiles_iter)
    return [standardize_one(s, chosen) for s in smiles_iter]


def standardize_column(df: pl.DataFrame, column: str = "SMILES", mode: Optional[str] = None) -> pl.DataFrame:
    """Return df with `column` standardized. A frame without that column is returned unchanged."""
    chosen = resolve(mode)
    if chosen == "none" or column not in df.columns:
        return df
    return df.with_columns(pl.Series(column, standardize_many(df[column].to_list(), chosen)))


def changed(original: Iterable[str], standardized: Iterable[str]) -> int:
    """How many SMILES the standardizer rewrote: what a caller reports to the user."""
    return sum(1 for a, b in zip(original, standardized) if a != b)


def add_argument(parser) -> None:
    """Add `--standardization` to an argparse parser, so every CLI spells it the same way."""
    parser.add_argument(
        "--standardization",
        default=DEFAULT_MODE,
        choices=list(MODES),
        help=(
            f"How SMILES are standardized before featurization (default: {DEFAULT_MODE}). "
            "'none' passes them through; 'normal' takes the largest fragment and canonicalizes, "
            "which is what every recorded result was produced under; 'advanced' runs RDKit's "
            "Cleanup + FragmentParent + Uncharger. Features are cached per SMILES string, so "
            "switching modes recomputes rather than reuses."
        ),
    )


def _mode_from_env() -> None:
    """Adopt the mode an outer process set, so subprocess backends match their parent."""
    from_env = os.environ.get(ENV_VAR)
    if from_env in _MODES:
        globals()["_mode"] = from_env


_mode_from_env()
