"""Structure standardization and identity keys for cross-source deduplication.

Every external source (ChEMBL, PubChem, BindingDB) writes SMILES its own way:
salts, tautomer choices, explicit stereo, charge states. Comparing raw SMILES
strings therefore under-counts overlap. We key molecules on the *connectivity
block* of the InChIKey -- the first 14 characters -- which is invariant to
stereochemistry, protonation and isotope labelling, so salt and stereo
variants of the same skeleton collapse to one key.
"""

from typing import Iterable, Optional

_SALT_STRIPPER = None


def _largest_fragment(smiles: str) -> str:
    """Return the largest covalent fragment of a multi-component SMILES.

    Counter-ions (Cl-, Na+, tosylate, ...) carry no CYP information, so a
    salt is reduced to its parent by taking the fragment with the most atoms.
    """
    if "." not in smiles:
        return smiles
    return max(smiles.split("."), key=len)


def standardize_smiles(smiles: str) -> Optional[str]:
    """Return a canonical, desalted, uncharged SMILES, or None if unparsable.

    Neutralization is left to RDKit's default sanitization plus fragment
    selection; we deliberately do not run a full tautomer canonicalizer,
    which is slow and, for a 31k-molecule merged table, changes few keys.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
        from rdkit import RDLogger  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - rdkit is a hard dep at runtime
        raise ImportError("rdkit is required for standardization") from exc

    RDLogger.DisableLog("rdApp.*")
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(_largest_fragment(smiles.strip()))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def connectivity_key(smiles: str) -> Optional[str]:
    """Return the InChIKey connectivity block (first 14 chars), or None.

    This is the join key for merging sources and for measuring train/test
    overlap: two rows share a key when they describe the same molecular
    skeleton regardless of salt form or stereochemistry.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
        from rdkit import RDLogger  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required for standardization") from exc

    RDLogger.DisableLog("rdApp.*")
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(_largest_fragment(smiles.strip()))
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    if not key:
        return None
    return key.split("-")[0]


def connectivity_keys(smiles_iter: Iterable[str]) -> list[Optional[str]]:
    """Vectorized connectivity_key over an iterable of SMILES."""
    return [connectivity_key(s) for s in smiles_iter]


def stereo_key(smiles: str) -> Optional[str]:
    """Return the full InChIKey (connectivity + stereo block), or None.

    Use this, not connectivity_key, to decide whether two rows of the *same*
    source are the same measurement. Stereoisomers share a connectivity block
    but are different molecules with different CYP labels: quinidine and
    quinine (LOUPRKONTZGTKE) differ by 2.56 log units on CYP2D6. Collapsing
    on connectivity block would average that away.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
        from rdkit import RDLogger  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rdkit is required for standardization") from exc

    RDLogger.DisableLog("rdApp.*")
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(_largest_fragment(smiles.strip()))
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol) or None


def stereo_keys(smiles_iter: Iterable[str]) -> list[Optional[str]]:
    """Vectorized stereo_key over an iterable of SMILES."""
    return [stereo_key(s) for s in smiles_iter]
