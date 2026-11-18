"""The PXR challenge's own reshaping: how its CSVs are spelled, and the derive steps that normalise them.

Private to this group, so the shared `data_tools.fetchers` keeps only what more than one dataset family needs.

The source files name columns the way the assay report does: `Molecule Name` with a space, units in
parentheses (`Emax_estimate (log2FC vs. baseline)`). The rest of the project keys on `Molecule_Name` and
selects label columns by name, so every derive step here applies one mapping. Columns it does not name are
copied through, and an entry for a column a file lacks is ignored, so the same mapping serves the train
split, the blinded test set, the released labels and the structure track alike.
"""

import sys
from pathlib import Path

from data_tools.dataset_spec import Step

REPO = "hf://datasets/openadmet/pxr-challenge-train-test"

RENAMES = {
    "Molecule Name": "Molecule_Name",
    "OCNT Batch": "OCNT_Batch",
    "smiles": "SMILES",  # the structure track lower-cases it
    # pEC50, the scored label, with its uncertainty
    "pEC50_std.error (-log10(molarity))": "pEC50_std_error",
    "pEC50_ci.lower (-log10(molarity))": "pEC50_ci_lower",
    "pEC50_ci.upper (-log10(molarity))": "pEC50_ci_upper",
    # Emax: how far induction goes at saturation, as a log2 fold change over baseline
    "Emax_estimate (log2FC vs. baseline)": "Emax_log2FC",
    "Emax_std.error (log2FC vs. baseline)": "Emax_log2FC_std_error",
    "Emax_ci.lower (log2FC vs. baseline)": "Emax_log2FC_ci_lower",
    "Emax_ci.upper (log2FC vs. baseline)": "Emax_log2FC_ci_upper",
    # the same ceiling against the positive control (rifampicin), so 1.0 is a full inducer
    "Emax.vs.pos.ctrl_estimate (dimensionless)": "Emax_vs_pos_ctrl",
    "Emax.vs.pos.ctrl_std.error (dimensionless)": "Emax_vs_pos_ctrl_std_error",
    "Emax.vs.pos.ctrl_ci.lower (dimensionless)": "Emax_vs_pos_ctrl_ci_lower",
    "Emax.vs.pos.ctrl_ci.upper (dimensionless)": "Emax_vs_pos_ctrl_ci_upper",
    # the challenge's own Train/Test marker; the pinned modelling split is computed, not read from here
    "Split": "challenge_split",
}


def _read(path: Path):
    import polars as pl

    # infer_schema_length=None: sparse columns would otherwise be typed from the first rows alone.
    frame = pl.read_csv(path, infer_schema_length=None)
    return frame.rename({old: new for old, new in RENAMES.items() if old in frame.columns})


def _write(frame, path: Path) -> None:
    frame.write_csv(path)
    print(f"Wrote {len(frame)} rows -> {path}", file=sys.stderr)


def normalise(source: str, dest: str, keep: tuple[str, ...] = ()) -> Step:
    """A derive step that renames `source`'s columns into `dest`, optionally keeping only some of them."""

    def derive(data_dir: Path) -> None:
        frame = _read(data_dir / source)
        if keep:
            frame = frame.select([c for c in keep if c in frame.columns])
        _write(frame, data_dir / dest)

    return derive


def stack(sources: tuple[str, ...], dest: str) -> Step:
    """A derive step that renames and stacks several CSVs into `dest`.

    The test labels were released one phase at a time, so one logical split arrives as two files. Rows keep
    the order given; a column missing from one file is null there.
    """

    def derive(data_dir: Path) -> None:
        import polars as pl

        frames = [_read(data_dir / name) for name in sources]
        _write(pl.concat(frames, how="diagonal_relaxed"), data_dir / dest)

    return derive
