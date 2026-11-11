"""Test validation predictions, seed averaging, model filters, and clustering."""

import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import models.validation as validation
from models.validation import (
    validation_prediction,
)

@pytest.mark.parametrize("shared_block", [False, True])
def test_validation_predictions_and_truth(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch,
    shared_block: bool,
): 
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "split": ["train", "train", "train", "val", "val"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5]
            for target in targets
        },
    })

    if shared_block:
        df = df.with_columns(
            pl.when(pl.col("split") == "val")
            .then(pl.lit("VAL_SHARED"))
            .otherwise(pl.col("inchikey_block"))
            .alias("inchikey_block")
        )

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(validation, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(validation.REGISTRY, "fake_model", FakeModel)

    result, truth = validation_prediction(
        df=df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        cache_dir=tmp_path / "features",
    )

    assert result.height == 8
    assert set(result.columns) == {
        "model_id", "inchikey_full", "target", "pred"
    }
    assert result["model_id"].unique().to_list() == ["fake_model_test_seed42"]
    assert result["pred"].dtype == pl.Float64

    expected_keys = {
        (molecule, target)
        for molecule in ["VAL_A-full", "VAL_B-full"]
        for target in targets
    }
    actual_keys = set(
        result.select("inchikey_full", "target").iter_rows()
    )
    assert actual_keys == expected_keys

    for target in targets:
        rows = result.filter(pl.col("target") == target).sort("inchikey_full")
        np.testing.assert_array_equal(rows["pred"].to_numpy(), [4.0, 5.0])

    # Four distinct models, each fitted on training rows only.
    assert len(instances) == 4
    for model in instances:
        np.testing.assert_array_equal(model.X_train, [[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(model.y_train, [5.0, 6.0, 7.0])

    assert truth.height == 8
    assert set(truth.columns) == {"inchikey_full", "target", "truth"}
    assert set(truth.select("inchikey_full", "target").iter_rows()) == expected_keys

    for target in targets:
        rows = truth.filter(pl.col("target") == target).sort("inchikey_full")
        np.testing.assert_array_equal(rows["truth"].to_numpy(), [5.5, 6.5])

def test_validation_respects_pinned_split(tmp_path, monkeypatch):
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full", "TEST_A-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B", "TEST_A"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
        "split": ["train", "train", "train", "val", "val", "test"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5, 8.0]
            for target in targets
        },
    })

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(validation, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            self.X_val = X.copy()
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(validation.REGISTRY, "fake_model", FakeModel)

    result, truth = validation_prediction(
        df=df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        cache_dir=tmp_path / "features",
    )

    assert len(instances) == 4

    for model in instances:
        np.testing.assert_array_equal(model.X_train, [[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(model.y_train, [5.0, 6.0, 7.0])
        np.testing.assert_array_equal(model.X_val, [[4.0], [5.0]])

    assert set(result["inchikey_full"]) == {"VAL_A-full", "VAL_B-full"}
    assert set(truth["inchikey_full"]) == {"VAL_A-full", "VAL_B-full"}
      
def test_validation_handles_missing_labels(tmp_path, monkeypatch):
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "split": ["train", "train", "train", "val", "val"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5]
            for target in targets
        },
    })

    df = df.with_columns(
        pl.when(pl.col("inchikey_block") == "TRAIN_A")
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(targets[0]))   
        .alias(targets[0])  
        )
    df = df.with_columns(
        pl.when(pl.col("inchikey_block") == "VAL_A")
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(targets[0]))     
        .alias(targets[0])  
        )

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(validation, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            self.X_val = X.copy()
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(validation.REGISTRY, "fake_model", FakeModel)

    result, truth = validation_prediction(
        df=df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        cache_dir=tmp_path / "features",
    )

    assert len(instances) == len(targets)

    for target, model in zip(targets, instances, strict=True):
        if target == targets[0]:
            expected_X_train = [[2.0], [3.0]]
            expected_y_train = [6.0, 7.0]
            expected_X_val = [[5.0]]
        else:
            expected_X_train = [[1.0], [2.0], [3.0]]
            expected_y_train = [5.0, 6.0, 7.0]
            expected_X_val = [[4.0], [5.0]]

        np.testing.assert_array_equal(model.X_train, expected_X_train)
        np.testing.assert_array_equal(model.y_train, expected_y_train)
        np.testing.assert_array_equal(model.X_val, expected_X_val)

    expected_keys = {
        (molecule, target)
        for molecule in ["VAL_A-full", "VAL_B-full"]
        for target in targets
    }
    expected_keys.remove(("VAL_A-full", targets[0]))

    assert result.height == truth.height == 7
    assert set(truth.select("inchikey_full", "target").iter_rows()) == expected_keys

    for target in targets:
        rows = truth.filter(pl.col("target") == target).sort("inchikey_full")
        expected = [6.5] if target == targets[0] else [5.5, 6.5]
        np.testing.assert_array_equal(rows["truth"].to_numpy(), expected)

@pytest.mark.parametrize(
    "case, message",
    [
        ("missing_split", "split column must exist"),
        ("missing_column", "data missing columns"),
        ("empty_train", "training data is empty"),
        ("empty_val", "validation data is empty"),
        ("duplicate_id", "duplicate molecule IDs"),
        ("overlap", "Training and validation molecules overlap"),
        ("unknown_model", "Unknown model"),
        ("unknown_feature", "Unknown input"),
        ("empty_features", "At least one feature representation"),
        ("empty_targets", "Targets are empty"),
        ("duplicate_targets", "Duplicate target names"),
        ("unexpected_target", "Unexpected targets"),
        ("missing_target", "missing scored targets"),
        ("invalid_model_id", "model_id must be a nonblank string"),
        ("invalid_params", "model_params must be dict"),
    ],
)
def test_validation_rejects_invalid_inputs(
    tmp_path, monkeypatch, case, message
):  
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "split": ["train", "train", "train", "val", "val"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5]
            for target in targets
        },
    })

    model_name = "decision_tree"
    model_id = "validation_test"
    model_params = {}
    input_names = ["rdkit"]
    targets = targets.copy()

    if case == "missing_split":
        df = df.drop("split")

    elif case == "missing_column":
        df = df.drop("SMILES")

    elif case == "empty_train":
        df = df.filter(pl.col("split") != "train")

    elif case == "empty_val":
        df = df.filter(pl.col("split") != "val")

    elif case == "duplicate_id":
        duplicate = df.filter(pl.col("split") == "train").head(1)
        df = pl.concat([df, duplicate])

    elif case == "overlap":
        df = df.with_columns(
            pl.when(pl.col("inchikey_block") == "VAL_A")
            .then(pl.lit("TRAIN_A"))
            .otherwise(pl.col("inchikey_block"))
            .alias("inchikey_block")
        )

    elif case == "unknown_model":
        model_name = "unknown_model"

    elif case == "unknown_feature":
        input_names = ["unknown_feature"]

    elif case == "empty_features":
        input_names = []

    elif case == "empty_targets":
        targets = []

    elif case == "duplicate_targets":
        targets.append(targets[0])

    elif case == "unexpected_target":
        targets.append("unexpected_target")
        df = df.with_columns(pl.lit(1.0).alias("unexpected_target"))

    elif case == "missing_target":
        targets = targets[:-1]

    elif case == "invalid_model_id":
        model_id = ""

    elif case == "invalid_params":
        model_params = []

    else:
        pytest.fail(f"Unhandled case: {case}")

    with pytest.raises(ValueError, match=message):
        validation_prediction(
            df=df,
            model_name=model_name,
            model_id=model_id,
            model_params=model_params,
            input_names=input_names,
            targets=targets,
            cache_dir=tmp_path / "features",
        )

@pytest.mark.parametrize(
    "case, message",
    [
        ("one_dimensional", "features must be a 2D matrix"),
        ("wrong_row_count", "feature row count does not match"),
        ("zero_columns", "features have no columns"),
        ("different_widths", "feature widths differ"),
        ("val_nan", "Non-finite val features"),
        ("val_inf", "Non-finite val features"),
        ("val_negative_inf", "Non-finite val features"),
    ],
)
def test_validation_rejects_invalid_features(
    tmp_path, monkeypatch, case, message
):
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "split": ["train", "train", "train", "val", "val"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5]
            for target in targets
        },
    })

    def fake_featurize(df, input_names, cache_dir):
        features = np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )
        # Keep training features valid; corrupt validation features only.
        if df["split"][0] == "train":
            return features

        if case == "one_dimensional":
            return features[:, 0]
        elif case == "wrong_row_count":
            return features[:-1]
        elif case == "zero_columns":
            return features[:, :0]
        elif case == "different_widths":
            return np.column_stack([features, features])
        elif case == "val_nan":
            features[0, 0] = np.nan
        elif case == "val_inf":
            features[0, 0] = np.inf
        elif case == "val_negative_inf":
            features[0, 0] = -np.inf
        else:
            pytest.fail(f"Unhandled case: {case}")

        return features

    monkeypatch.setattr(validation, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(validation.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match=message):
        validation_prediction(
            df=df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            cache_dir=tmp_path / "features",
        )

@pytest.mark.parametrize(
    "bad_output, message",
    [
        (np.array(4.0), "must be one-dimensional"),
        (np.array([[4.0], [5.0]]), "must be one-dimensional"),
        (np.array([4.0]), "Prediction count does not match"),
        (np.array([4.0, 5.0, 6.0]), "Prediction count does not match"),
        (np.array([4.0, np.nan]), "contain non-finite values"),
        (np.array([4.0, np.inf]), "contain non-finite values"),
        (np.array([4.0, -np.inf]), "contain non-finite values"),
    ],
)
def test_validation_rejects_invalid_predictions(
    tmp_path, monkeypatch, bad_output, message
):
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    df = pl.DataFrame({
        "inchikey_full": [
            "TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full",
            "VAL_A-full", "VAL_B-full",
        ],
        "inchikey_block": [
            "TRAIN_A", "TRAIN_B", "TRAIN_C", "VAL_A", "VAL_B"
        ],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
        "split": ["train", "train", "train", "val", "val"],
        **{
            target: [5.0, 6.0, 7.0, 5.5, 6.5]
            for target in targets
        },
    })

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(validation, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return bad_output
            
    monkeypatch.setitem(validation.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match=message):
        validation_prediction(
            df=df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            cache_dir=tmp_path / "features",
        )


@pytest.fixture
def validation_seed_setup(tmp_path, monkeypatch):
    targets = list(validation._SCORED_DEFAULT)

    df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A", "VAL_A", "VAL_B"],
        "inchikey_block": ["TRAIN_A", "VAL_A", "VAL_B"],
        "SMILES": ["C", "CC", "CCC"],
        "split": ["train", "val", "val"],
        **{target: [5.0, 6.0, 7.0] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": molecule,
            "target": target,
            "truth": value,
        }
        for molecule, value in [("VAL_A", 6.0), ("VAL_B", 7.0)]
        for target in targets
    ])

    kwargs = {
        "df": df,
        "model_name": "random_forest",
        "model_id": "example",
        "model_params": {"n_estimators": 10, "random_state": 999},
        # The backend is mocked, so this name is not featurized.
        "input_names": ["test_features"],
        "output_dir": tmp_path / "runs",
        "cache_dir": tmp_path / "features",
        "seeds": [11, 22, 33],
        "targets": targets,
    }

    calls = []
    returned_predictions = {}
    state = {"truth_case": "same"}

    def fake_run(**call):
        calls.append(call)
        seed = call["model_params"]["random_state"]

        predictions = truth.select("inchikey_full", "target").with_columns(
            pl.lit(call["model_id"]).alias("model_id"),
            pl.lit(float(seed)).alias("pred"),
        )
        returned_predictions[seed] = predictions

        run_truth = truth.clone()
        if len(calls) > 1:
            if state["truth_case"] == "shuffled":
                run_truth = run_truth.reverse()
            elif state["truth_case"] == "changed":
                run_truth = run_truth.with_columns(
                    (pl.col("truth") + 1.0).alias("truth")
                )

        return predictions, run_truth

    monkeypatch.setattr(validation, "validation_prediction", fake_run)

    return kwargs, calls, truth, returned_predictions, state

@pytest.mark.parametrize("model_name", ["random_forest"])
def test_validation_seeds_runs_each_seed(validation_seed_setup, model_name):
    kwargs, calls, expected_truth, returned_predictions, _ = validation_seed_setup
    kwargs["model_name"] = model_name

    predictions, truth = validation.run_validation_seeds(**kwargs)

    assert len(calls) == 3
    assert set(predictions) == set(kwargs["seeds"])

    for seed, call in zip(kwargs["seeds"], calls, strict=True):
        assert call["df"] is kwargs["df"]
        assert call["model_id"] == f"example_seed{seed}"
        assert_frame_equal(predictions[seed], returned_predictions[seed])

        assert call["model_name"] == model_name
        assert call["model_params"] == {
            "n_estimators": 10,
            "random_state": seed,
        }
        assert call["input_names"] == kwargs["input_names"]
        assert call["targets"] == kwargs["targets"]
        assert call["cache_dir"] == kwargs["cache_dir"]

    keys = ["inchikey_full", "target"]
    assert_frame_equal(truth, expected_truth.sort(keys))

@pytest.mark.parametrize("model_name", ["random_forest"])
def test_validation_seeds_preserves_configs(validation_seed_setup, model_name):
    kwargs, calls, _, _, _ = validation_seed_setup
    kwargs["model_name"] = model_name

    original_params = deepcopy(kwargs["model_params"])

    validation.run_validation_seeds(**kwargs)

    assert kwargs["model_params"] == original_params

    if model_name == "random_forest":
        copies = [call["model_params"] for call in calls]
        assert all(params is not kwargs["model_params"] for params in copies)
        assert len({id(params) for params in copies}) == 3

@pytest.mark.parametrize(
    "model_name, overrides, match",
    [
        ("random_forest", {"seeds": [1, 2]}, "at least 3"),
        ("random_forest", {"seeds": [1, 1, 2]}, "unique"),
        ("random_forest", {"seeds": [1, 2, True]}, "integers"),
        ("random_forest", {"seeds": [1, 2, 3.0]}, "integers"),
        ("random_forest", {"seeds": [1, 2, "3"]}, "integers"),
        ("random_forest", {"model_id": " "}, "nonblank string"),
        ("random_forest", {"input_names": None}, "input_names are required"),
        ("random_forest", {"cache_dir": None}, "cache_dir is required"),
        ("random_forest", {"model_params": []}, "model_params must be dict"),
    ],
)
def test_validation_seeds_rejects_invalid_inputs(
    validation_seed_setup,
    model_name,
    overrides,
    match,
):
    kwargs, calls, _, _, _ = validation_seed_setup
    kwargs["model_name"] = model_name
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        validation.run_validation_seeds(**kwargs)

    assert calls == []

@pytest.mark.parametrize("model_name", ["random_forest"])
@pytest.mark.parametrize("truth_case", ["shuffled", "changed"])
def test_validation_seeds_checks_truth(
    validation_seed_setup,
    model_name,
    truth_case,
):
    kwargs, calls, expected_truth, _, state = validation_seed_setup
    kwargs["model_name"] = model_name
    state["truth_case"] = truth_case

    if truth_case == "changed":
        with pytest.raises(ValueError, match="truth is different"):
            validation.run_validation_seeds(**kwargs)

        assert len(calls) == 2
    else:
        _, truth = validation.run_validation_seeds(**kwargs)

        assert len(calls) == 3
        assert_frame_equal(
            truth,
            expected_truth.sort(["inchikey_full", "target"]),
        ) 

@pytest.fixture
def validation_seed_tables():
    truth = pl.DataFrame({
        "inchikey_full": ["A", "B", "A", "B"],
        "target": ["target_1", "target_1", "target_2", "target_2"],
        "truth": [1.0, 2.0, 3.0, 4.0],
    })

    predictions = {
        seed: truth.select("inchikey_full", "target").with_columns(
            pl.lit(f"model_seed{seed}").alias("model_id"),
            pl.Series("pred", [value + offset for value in [1., 2., 3., 4.]]),
        )
        for seed, offset in enumerate([-1.0, 0.0, 1.0])
    }
    return predictions, truth


def test_validation_seed_average_values(validation_seed_tables):
    predictions, truth = validation_seed_tables

    result = validation._seed_averaging(predictions, truth, "averaged")

    expected = truth.rename({"truth": "pred"}).with_columns(
        pl.lit("averaged").alias("model_id")
    ).select("model_id", "inchikey_full", "target", "pred")

    assert_frame_equal(
        result,
        expected.sort(["inchikey_full", "target"]),
    )


def test_validation_seed_average_ignores_row_order(validation_seed_tables):
    predictions, truth = validation_seed_tables
    expected = validation._seed_averaging(predictions, truth, "averaged")

    shuffled = {
        seed: table.sample(fraction=1.0, shuffle=True, seed=seed + 10)
        for seed, table in predictions.items()
    }

    result = validation._seed_averaging(
        shuffled,
        truth.reverse(),
        "averaged",
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "case, match",
    [
        ("too_few_seeds", "at least three"),
        ("invalid_seed", "seed keys must be integers"),
        ("blank_model_id", "model_id cannot be blank"),
        ("missing_column", "Missing required columns"),
        ("multiple_models", "exactly one model_id"),
        ("null_pred", "pred cannot be null"),
        ("nan_pred", "pred must be finite"),
        ("infinite_pred", "pred must be finite"),
        ("nonnumeric_pred", "pred must be numeric"),
        ("empty_truth", "truth cannot be empty"),
        ("duplicate_truth", "duplicates in truth"),
    ],
)
def test_validation_seed_average_rejects_invalid_inputs(
    validation_seed_tables,
    case,
    match,
):
    predictions, truth = validation_seed_tables
    model_id = "averaged"

    if case == "too_few_seeds":
        predictions.pop(0)
    elif case == "invalid_seed":
        predictions["bad"] = predictions.pop(0)
    elif case == "blank_model_id":
        model_id = " "
    elif case == "missing_column":
        predictions[0] = predictions[0].drop("pred")
    elif case == "multiple_models":
        predictions[0] = predictions[0].with_columns(
            pl.Series("model_id", ["a", "a", "b", "b"])
        )
    elif case == "nonnumeric_pred":
        predictions[0] = predictions[0].with_columns(
            pl.lit("bad").alias("pred")
        )
    elif case in {"null_pred", "nan_pred", "infinite_pred"}:
        value = {
            "null_pred": None,
            "nan_pred": float("nan"),
            "infinite_pred": float("inf"),
        }[case]
        predictions[0] = predictions[0].with_columns(
            pl.lit(value, dtype=pl.Float64).alias("pred")
        )
    elif case == "empty_truth":
        truth = truth.head(0)
    elif case == "duplicate_truth":
        truth = pl.concat([truth, truth.head(1)])

    with pytest.raises(ValueError, match=match):
        validation._seed_averaging(predictions, truth, model_id)


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing", "missing truth keys"),
        ("extra", "extra keys"),
        ("duplicate", "duplicates in prediction"),
    ],
)
def test_validation_seed_average_rejects_mismatched_keys(
    validation_seed_tables,
    case,
    match,
):
    predictions, truth = validation_seed_tables
    table = predictions[0]

    if case == "missing":
        table = table.slice(1)
    elif case == "extra":
        extra = table.head(1).with_columns(
            pl.lit("EXTRA").alias("inchikey_full")
        )
        table = pl.concat([table, extra])
    elif case == "duplicate":
        table = pl.concat([table, table.head(1)])

    predictions[0] = table

    with pytest.raises(ValueError, match=match):
        validation._seed_averaging(predictions, truth, "averaged")

@pytest.fixture
def r_squared_tables():
    truth_rows = []
    prediction_rows = []

    # Each target has truth variance denominator = 50.
    # Squared errors: 0 → R²=1; 35 → R²=0.3; 36 → R²=0.28.
    errors = {
        "target_1": {
            "model_a": [0.0, 0.0, 0.0],
            "model_b": [1.0, 3.0, 5.0],
            "model_c": [0.0, 0.0, 6.0],
        },
        "target_2": {
            "model_a": [0.0, 0.0, 6.0],
            "model_b": [0.0, 0.0, 0.0],
            "model_c": [1.0, 3.0, 5.0],
        },
    }

    for target, offset in [("target_1", 0.0), ("target_2", 10.0)]:
        for i, molecule in enumerate(["A", "B", "C"]):
            value = offset + 5.0 * i
            truth_rows.append({
                "inchikey_full": molecule,
                "target": target,
                "truth": value,
            })

            for model_id, model_errors in errors[target].items():
                prediction_rows.append({
                    "model_id": model_id,
                    "inchikey_full": molecule,
                    "target": target,
                    "pred": value + model_errors[i],
                })

    return pl.DataFrame(prediction_rows), pl.DataFrame(truth_rows)


def test_r_squared_filter_scores_and_threshold(r_squared_tables):
    predictions, truth = r_squared_tables
    result = validation.r_squared_filtering(predictions, truth)

    assert result.schema["keep"] == pl.Boolean

    expected = {
        ("model_a", "target_1"): (1.0, True),
        ("model_b", "target_1"): (0.3, True),
        ("model_c", "target_1"): (0.28, False),
        ("model_a", "target_2"): (0.28, False),
        ("model_b", "target_2"): (1.0, True),
        ("model_c", "target_2"): (0.3, True),
    }

    assert set(result.columns) == {
        "model_id", "target", "r_squared", "keep"
    }

    actual = {
        (row["model_id"], row["target"]): row
        for row in result.iter_rows(named=True)
    }
    assert result.height == len(actual) == len(expected)
    assert actual.keys() == expected.keys()

    for key, (score, keep) in expected.items():
        assert actual[key]["r_squared"] == pytest.approx(score)
        assert actual[key]["keep"] == keep


def test_r_squared_filter_ignores_row_order(r_squared_tables):
    predictions, truth = r_squared_tables
    expected = validation.r_squared_filtering(predictions, truth)

    result = validation.r_squared_filtering(
        predictions.sample(fraction=1.0, shuffle=True, seed=10),
        truth.sample(fraction=1.0, shuffle=True, seed=20),
    )

    keys = ["model_id", "target"]
    assert_frame_equal(result.sort(keys), expected.sort(keys))


@pytest.mark.parametrize("case", ["missing", "duplicate"])
def test_r_squared_filter_rejects_missing_or_duplicate_predictions(
    r_squared_tables,
    case,
):
    predictions, truth = r_squared_tables

    if case == "missing":
        predictions = predictions.slice(1)
        error = ValueError
        match = "Missing predictions"
    else:
        predictions = pl.concat([predictions, predictions.head(1)])
        # The join's validate="1:1" raises a Polars error.
        error = pl.exceptions.ComputeError
        match = "1:1"

    with pytest.raises(error, match=match):
        validation.r_squared_filtering(predictions, truth)


@pytest.mark.parametrize("table_name", ["predictions", "truth"])
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf")],
    ids=["null", "nan", "infinity"],
)
def test_r_squared_filter_rejects_invalid_values(
    r_squared_tables,
    table_name,
    value,
):
    predictions, truth = r_squared_tables

    if table_name == "predictions":
        predictions = predictions.with_columns(
            pl.lit(value, dtype=pl.Float64).alias("pred")
        )
        match = "Missing predictions" if value is None else "must be finite"
    else:
        truth = truth.with_columns(
            pl.lit(value, dtype=pl.Float64).alias("truth")
        )
        match = "missing labels" if value is None else "must be finite"

    with pytest.raises(ValueError, match=match):
        validation.r_squared_filtering(predictions, truth)


@pytest.mark.parametrize("case", ["single_label", "constant_labels"])
def test_r_squared_filter_rejects_undefined_scores(r_squared_tables, case):
    predictions, truth = r_squared_tables

    if case == "single_label":
        truth = truth.filter(pl.col("inchikey_full") == "A")
        predictions = predictions.filter(pl.col("inchikey_full") == "A")
    else:
        truth = truth.with_columns(pl.lit(5.0).alias("truth"))

    with pytest.raises(ValueError, match="at least two labels with nonzero variance"):
        validation.r_squared_filtering(predictions, truth)


@pytest.fixture
def rae_tables():
    validation_rows = []
    oof_rows = []
    validation_predictions = []
    oof_predictions = []

    ratios = {
        "target_1": {"a": 1.0, "b": 2.0, "c": 3.0},
        "target_2": {"a": 3.0, "b": 1.0, "c": 2.0},
    }

    for target, offset in [("target_1", 0.0), ("target_2", 100.0)]:
        for i in range(3):
            # Different populations, label means, and label spreads.
            val_row = {
                "inchikey_full": f"VAL_{i}",
                "target": target,
                "truth": offset + 2.0 * i,
            }
            oof_row = {
                "inchikey_full": f"TRAIN_{i}",
                "fold": i,
                "target": target,
                "truth": offset + 10.0 + 4.0 * i,
            }
            validation_rows.append(val_row)
            oof_rows.append(oof_row)

            for model_id, ratio in ratios[target].items():
                validation_predictions.append({
                    "model_id": model_id,
                    "inchikey_full": val_row["inchikey_full"],
                    "target": target,
                    "pred": val_row["truth"] + ratio,
                })
                oof_predictions.append({
                    "model_id": model_id,
                    "inchikey_full": oof_row["inchikey_full"],
                    "fold": i,
                    "target": target,
                    "pred": oof_row["truth"] + 2.0,
                })

    return {
        "seed_averaged_validation_predictions": pl.DataFrame(
            validation_predictions
        ),
        "validation_truth": pl.DataFrame(validation_rows),
        "seed_averaged_oof_predictions": pl.DataFrame(oof_predictions),
        "oof_truth": pl.DataFrame(oof_rows),
    }


def test_rae_filter_scores_and_threshold(rae_tables):
    result = validation.heldout_rae_filter(**rae_tables)

    assert result.schema["keep"] == pl.Boolean

    expected_ratios = {
        ("a", "target_1"): 1.0,
        ("b", "target_1"): 2.0,
        ("c", "target_1"): 3.0,
        ("a", "target_2"): 3.0,
        ("b", "target_2"): 1.0,
        ("c", "target_2"): 2.0,
    }

    actual = {
        (row["model_id"], row["target"]): row
        for row in result.iter_rows(named=True)
    }

    assert result.height == len(actual) == 6
    assert actual.keys() == expected_ratios.keys()

    for key, ratio in expected_ratios.items():
        row = actual[key]
        assert row["oof_rae"] == pytest.approx(0.75)
        assert row["heldout_rae"] == pytest.approx(0.75 * ratio)
        assert row["rae_ratio"] == pytest.approx(ratio)
        assert row["keep"] == (ratio <= 2.0)


def test_rae_filter_ignores_row_order(rae_tables):
    expected = validation.heldout_rae_filter(**rae_tables)

    shuffled = {
        name: table.sample(fraction=1.0, shuffle=True, seed=i + 10)
        for i, (name, table) in enumerate(rae_tables.items())
    }
    result = validation.heldout_rae_filter(**shuffled)

    keys = ["model_id", "target"]
    assert_frame_equal(result.sort(keys), expected.sort(keys))


@pytest.mark.parametrize(
    "population, case",
    [
        ("validation", "missing"),
        ("validation", "duplicate"),
        ("oof", "missing"),
        ("oof", "duplicate"),
        ("oof", "wrong_fold"),
    ],
)
def test_rae_filter_rejects_bad_alignment(rae_tables, population, case):
    name = f"seed_averaged_{population}_predictions"
    table = rae_tables[name]

    if case == "missing":
        table = table.slice(1)
    elif case == "duplicate":
        table = pl.concat([table, table.head(1)])
    else:
        table = table.with_columns((pl.col("fold") + 10).alias("fold"))

    rae_tables[name] = table

    if case == "duplicate":
        error, match = pl.exceptions.ComputeError, "1:1"
    else:
        error, match = ValueError, "Missing predictions"

    with pytest.raises(error, match=match):
        validation.heldout_rae_filter(**rae_tables)


@pytest.mark.parametrize(
    "table_name, column",
    [
        ("validation_truth", "truth"),
        ("oof_truth", "truth"),
        ("seed_averaged_validation_predictions", "pred"),
        ("seed_averaged_oof_predictions", "pred"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf")],
    ids=["null", "nan", "infinity"],
)
def test_rae_filter_rejects_invalid_values(
    rae_tables,
    table_name,
    column,
    value,
):
    rae_tables[table_name] = rae_tables[table_name].with_columns(
        pl.lit(value, dtype=pl.Float64).alias(column)
    )

    if value is None:
        match = "missing labels" if column == "truth" else "Missing predictions"
    else:
        match = "must be finite"

    with pytest.raises(ValueError, match=match):
        validation.heldout_rae_filter(**rae_tables)


@pytest.mark.parametrize("population", ["validation", "oof"])
@pytest.mark.parametrize("case", ["single_label", "constant_labels"])
def test_rae_filter_rejects_undefined_rae(rae_tables, population, case):
    name = f"{population}_truth"

    if case == "single_label":
        # Keep one observed label per target.
        first_id = "VAL_0" if population == "validation" else "TRAIN_0"
        rae_tables[name] = rae_tables[name].filter(
            pl.col("inchikey_full") == first_id
        )
    else:
        rae_tables[name] = rae_tables[name].with_columns(
            pl.lit(5.0).alias("truth")
        )

    with pytest.raises(ValueError, match="at least two labels"):
        validation.heldout_rae_filter(**rae_tables)


def test_rae_filter_rejects_zero_oof_rae(rae_tables):
    keys = ["inchikey_full", "fold", "target"]
    name = "seed_averaged_oof_predictions"

    # Replace predictions with exact OOF truth values.
    rae_tables[name] = (
        rae_tables[name]
        .drop("pred")
        .join(rae_tables["oof_truth"], on=keys, validate="m:1")
        .rename({"truth": "pred"})
    )

    with pytest.raises(ValueError, match="oof_rae is 0"):
        validation.heldout_rae_filter(**rae_tables)


@pytest.fixture
def dedup_tables():
    # A and B correlate perfectly; C is negatively correlated.
    val = {
        "a": [0., 1., 2., 3.],
        "b": [2., 4., 6., 8.],
        "c": [3., 2., 1., 0.],
    }
    # OOF A and B are negatively correlated: clustering must still use val.
    oof = {
        "a": [0., 1., 2., 3.],
        "b": [8., 6., 4., 2.],
        "c": [0., 1., 2., 3.],
    }

    def predictions(values, prefix, with_fold=False):
        return pl.DataFrame([
            {
                "model_id": model,
                "inchikey_full": f"{prefix}{i}",
                "target": "t1",
                "pred": value,
                **({"fold": i % 2} if with_fold else {}),
            }
            for model, vector in values.items()
            for i, value in enumerate(vector)
        ])

    def truth(prefix, with_fold=False):
        return pl.DataFrame([
            {
                "inchikey_full": f"{prefix}{i}",
                "target": "t1",
                "truth": float(i),
                **({"fold": i % 2} if with_fold else {}),
            }
            for i in range(4)
        ])

    return {
        "seed_averaged_validation_predictions": predictions(val, "V"),
        "validation_truth": truth("V"),
        "seed_averaged_oof_predictions": predictions(oof, "O", True),
        "oof_truth": truth("O", True),
    }


def _dedup_groups(membership):
    return {
        frozenset(group["model_id"].to_list())
        for group in membership.partition_by(["target", "cluster_id"])
    }


def test_dedup_averages_both_populations_and_records_members(dedup_tables):
    val, oof, membership = validation.deduplication_filter(**dedup_tables)

    assert membership.height == 3
    assert _dedup_groups(membership) == {frozenset({"a", "b"}), frozenset({"c"})}

    for output, source_name, keys in [
        (val, "seed_averaged_validation_predictions", ["inchikey_full", "target"]),
        (oof, "seed_averaged_oof_predictions", ["inchikey_full", "fold", "target"]),
    ]:
        source = dedup_tables[source_name]
        expected = (
            source.join(
                membership, on=["model_id", "target"], validate="m:1"
            )
            .group_by(["cluster_id", *keys])
            .agg(pl.col("pred").mean())
            .rename({"cluster_id": "model_id"})
        )
        columns = ["model_id", *keys, "pred"]
        assert_frame_equal(
            output.select(columns).sort(columns[:-1]),
            expected.select(columns).sort(columns[:-1]),
        )


def test_dedup_uses_validation_clusters_for_oof(dedup_tables):
    _, oof, membership = validation.deduplication_filter(**dedup_tables)

    cluster = membership.filter(pl.col("model_id") == "a")["cluster_id"].item()
    assert membership.filter(pl.col("model_id") == "b")["cluster_id"].item() == cluster

    values = oof.filter(pl.col("model_id") == cluster).sort("inchikey_full")["pred"]
    np.testing.assert_allclose(values.to_numpy(), [4., 3.5, 3., 2.5])


def test_dedup_handles_different_models_per_target(dedup_tables):
    for name, table in list(dedup_tables.items()):
        extra = table.with_columns(pl.lit("t2").alias("target"))
        if "model_id" in extra.columns:
            extra = extra.filter(pl.col("model_id") == "a")
        dedup_tables[name] = pl.concat([table, extra])

    val, oof, membership = validation.deduplication_filter(**dedup_tables)

    for output in [val, oof]:
        assert set(output["target"]) == {"t1", "t2"}
    assert _dedup_groups(membership.filter(pl.col("target") == "t1")) == {
        frozenset({"a", "b"}), frozenset({"c"})
    }
    assert membership.filter(pl.col("target") == "t2")["model_id"].to_list() == ["a"]


def test_dedup_preserves_singletons(dedup_tables):
    for name in ["seed_averaged_validation_predictions", "seed_averaged_oof_predictions"]:
        dedup_tables[name] = dedup_tables[name].filter(pl.col("model_id") == "c")

    val, oof, membership = validation.deduplication_filter(**dedup_tables)
    assert _dedup_groups(membership) == {frozenset({"c"})}

    for output, name in [
        (val, "seed_averaged_validation_predictions"),
        (oof, "seed_averaged_oof_predictions"),
    ]:
        assert_frame_equal(
            output.drop("model_id").sort("inchikey_full"),
            dedup_tables[name].drop("model_id")
            .select(output.drop("model_id").columns)
            .sort("inchikey_full"),
        )


def test_dedup_ignores_row_order(dedup_tables):
    expected = validation.deduplication_filter(**dedup_tables)
    shuffled = {
        name: table.sample(fraction=1.0, shuffle=True, seed=i + 10)
        for i, (name, table) in enumerate(dedup_tables.items())
    }
    actual = validation.deduplication_filter(**shuffled)

    for left, right in zip(actual, expected, strict=True):
        assert_frame_equal(left.sort(left.columns), right.sort(right.columns))


def test_dedup_uses_complete_linkage(dedup_tables):
    # Centered, orthonormal vectors; adjacent angles correlate above .95,
    # but the two endpoints do not.
    u = np.array([-1., -1., 1., 1.]) / 2
    v = np.array([-1., 1., -1., 1.]) / 2
    rows = []
    for model, angle in zip(["a", "b", "c"], [0., 0.2, 0.4], strict=True):
        vector = np.cos(angle) * u + np.sin(angle) * v
        rows.extend({
            "model_id": model, "inchikey_full": f"V{i}",
            "target": "t1", "pred": float(value),
        } for i, value in enumerate(vector))

    dedup_tables["seed_averaged_validation_predictions"] = pl.DataFrame(rows)
    _, _, membership = validation.deduplication_filter(**dedup_tables)

    groups = _dedup_groups(membership)
    assert sorted(map(len, groups)) == [1, 2]
    assert not any({"a", "c"} <= group for group in groups)


@pytest.mark.parametrize("rho, merged", [(0.94, False), (0.95, False), (0.96, True)])
def test_dedup_uses_strict_threshold(dedup_tables, monkeypatch, rho, merged):
    for name in ["seed_averaged_validation_predictions", "seed_averaged_oof_predictions"]:
        dedup_tables[name] = dedup_tables[name].filter(pl.col("model_id") != "c")

    monkeypatch.setattr(
        validation.np, "corrcoef",
        lambda *args, **kwargs: np.array([[1., rho], [rho, 1.]]),
    )
    _, _, membership = validation.deduplication_filter(**dedup_tables)
    assert membership["cluster_id"].n_unique() == (1 if merged else 2)


@pytest.mark.parametrize(
    "case, error, match",
    [
        ("missing", ValueError, "Missing predictions"),
        ("duplicate", pl.exceptions.ComputeError, "1:1"),
        ("nonfinite", ValueError, "must be finite"),
        ("constant", ValueError, "undefined correlations"),
        ("no_models", ValueError, "No surviving models"),
    ],
)
def test_dedup_rejects_invalid_inputs(dedup_tables, case, error, match):
    name = "seed_averaged_validation_predictions"
    table = dedup_tables[name]

    if case == "missing":
        table = table.slice(1)
    elif case == "duplicate":
        table = pl.concat([table, table.head(1)])
    elif case == "nonfinite":
        table = table.with_columns(pl.lit(float("inf")).alias("pred"))
    elif case == "constant":
        table = table.with_columns(pl.lit(1.0).alias("pred"))
    else:
        table = table.head(0)

    dedup_tables[name] = table
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(error, match=match):
            validation.deduplication_filter(**dedup_tables)
