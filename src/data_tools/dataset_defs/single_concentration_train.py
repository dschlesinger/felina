"""Single-concentration screen: log2 fold change at 50 uM, one row per (molecule, enzyme).

This is the only in-house split that arrives long rather than wide, so it carries a `prepare` step that pivots it
before the head patterns are matched.
"""

from data_tools.dataset_spec import DatasetSpec, Source
from data_tools.fetchers import csv_from

URL = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-single-concentration-TRAIN.csv"
FILE = "single_concentration_train.csv"

_NAME = "Molecule_Name"


def _pivot(df):
    """Long (molecule, enzyme) rows to one `{enzyme}_log2fc` column per enzyme."""
    import polars as pl

    smiles = df.group_by(_NAME, maintain_order=True).agg(pl.col("SMILES").first())
    wide = df.pivot(on="enzyme", index=_NAME, values="log2fc_estimate", aggregate_function="mean")
    wide = wide.rename({c: f"{c}_log2fc" for c in wide.columns if c != _NAME})
    return smiles.join(wide, on=_NAME, how="left")


SPEC = DatasetSpec(
    name="single_concentration_train",
    group="challenge",
    description="Single-concentration train split",
    order=50,
    fetched=(FILE,),
    fetch=csv_from(URL, FILE),
    sources={
        "single_conc": Source(
            kind="inhouse",
            heads=[(r"^(.+)_log2fc$", "{}-SingleConc")],
            prepare=_pivot,
            description="log2 fold change at one concentration (50 uM)",
        ),
    },
)
