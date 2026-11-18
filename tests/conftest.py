"""Synthetic prediction tables built in memory for OOF and stacking tests."""

import polars as pl
import pytest

from data_tools.datasets import challenge


@pytest.fixture
def stack_blends() -> dict[str, tuple[float, float]]:
    """Return the model_a/model_b mixture behind each scored target's truth.

    Keyed on the challenge's own targets, so the tables built from it are the
    ones every validator expects without being told; a distinct blend each.
    """
    return {
        target: (0.2 * (index + 1), 1.0 - 0.2 * (index + 1))
        for index, target in enumerate(challenge().targets)
    }


@pytest.fixture
def stack_tables(stack_blends) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return fresh tables with known NNLS weights and a redundant model."""
    molecules = [
        ("SYNTHETIC_A", 0),
        ("SYNTHETIC_B", 0),
        ("SYNTHETIC_C", 1),
        ("SYNTHETIC_D", 1),
    ]
    # Each target's truth is a * model_a + b * model_b, so NNLS must recover (a, b, 0).
    target_values = {
        target: [a, b, 0.0, 1.0] for target, (a, b) in stack_blends.items()
    }
    model_values = {
        "model_a": [1.0, 0.0, 0.0, 1.0],
        "model_b": [0.0, 1.0, 0.0, 1.0],
        "model_c": [0.0, 0.0, 1.0, 1.0],
    }
    truth_rows = []
    prediction_rows = []
    for target, values in target_values.items():
        for (molecule, fold), value in zip(molecules, values, strict=True):
            truth_rows.append({
                "inchikey_full": molecule,
                "fold": fold,
                "target": target,
                "truth": value,
            })
        for model_id, predictions in model_values.items():
            for (molecule, fold), value in zip(molecules, predictions, strict=True):
                prediction_rows.append({
                    "model_id": model_id,
                    "inchikey_full": molecule,
                    "fold": fold,
                    "target": target,
                    "pred": value,
                })

    return pl.DataFrame(prediction_rows), pl.DataFrame(truth_rows)


@pytest.fixture
def oof_tables(stack_tables) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one model's held-out fold predictions and the complete truth."""
    predictions, truth = stack_tables
    predictions = predictions.filter(
        (pl.col("model_id") == "model_a") & (pl.col("fold") == 0)
    )
    return predictions, truth
