import numpy as np
import pytest

from models.recalibrate import apply_affine, fit_affine, recalibrate


def test_fit_recovers_known_map():
    p = np.linspace(4, 8, 50)
    slope, intercept = fit_affine(p, 1.7 * p - 3.0)
    assert slope == pytest.approx(1.7)
    assert intercept == pytest.approx(-3.0)
    assert apply_affine(p, slope, intercept) == pytest.approx(1.7 * p - 3.0)


def test_constant_predictions_give_identity():
    assert fit_affine(np.full(10, 5.0), np.arange(10.0)) == (1.0, 0.0)
    assert fit_affine(np.array([5.0]), np.array([1.0])) == (1.0, 0.0)


def test_recalibrate_fixes_compressed_predictions_and_holds_out_each_fold():
    rng = np.random.default_rng(0)
    y = rng.uniform(4, 8, 120)
    folds = np.arange(120) % 4
    raw = 0.5 * y + 2.0
    out = {
        "val": raw[:10], "blind": raw[10:15],
        "holdout": {k: raw[folds == k] for k in range(4)},
    }
    cal, (slope, intercept) = recalibrate(out, y, folds)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(-4.0)
    assert cal["val"] == pytest.approx(y[:10])
    assert cal["blind"] == pytest.approx(y[10:15])
    for k in range(4):
        assert cal["holdout"][k] == pytest.approx(y[folds == k])


def test_recalibrate_without_blind():
    y = np.arange(20.0)
    folds = np.arange(20) % 2
    out = {"val": y[:3], "blind": None, "holdout": {k: y[folds == k] for k in range(2)}}
    cal, _ = recalibrate(out, y, folds)
    assert cal["blind"] is None


def test_write_affine_csv(tmp_path):
    from models.evaluate import _AFFINE_COLUMNS, _write_affine

    rows = [dict(zip(_AFFINE_COLUMNS, ("CYP3A4_pIC50_direct_inhibition", "ridge", "maccs", "scaffold", 42, 1868, 0.75, 1.0)))]
    path = tmp_path / "nested" / "affine.csv"
    _write_affine(path, rows)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(_AFFINE_COLUMNS)
    assert lines[1] == "CYP3A4_pIC50_direct_inhibition,ridge,maccs,scaffold,42,1868,0.75,1.0"
    with pytest.raises(OSError, match="already exists|Cannot write"):
        _write_affine(path, rows)


def test_member_predictions_round_trip(tmp_path):
    """The saved member rows are enough to refit the run's affine map without refitting the models."""
    from models.evaluate import _MEMBER_COLUMNS, _write_members

    rng = np.random.default_rng(5)
    y = rng.normal(size=40)
    pred = 0.5 * y + 1.0 + 0.1 * rng.normal(size=40)
    folds = np.arange(40) % 4
    rows = [
        dict(zip(_MEMBER_COLUMNS, (f"C{i}", "CYP3A4_pIC50_direct_inhibition", "ridge", "maccs", "", int(folds[i]),
                                   "holdout", float(y[i]), float(pred[i]))))
        for i in range(40)
    ]
    path = tmp_path / "member_predictions.csv"
    _write_members(path, rows)

    import csv as _csv

    with path.open(encoding="utf-8") as stream:
        back = [r for r in _csv.DictReader(stream) if r["role"] == "holdout"]
    assert len(back) == 40
    got = fit_affine(np.array([float(r["y_pred"]) for r in back]), np.array([float(r["y_true"]) for r in back]))
    assert got == pytest.approx(fit_affine(pred, y))
