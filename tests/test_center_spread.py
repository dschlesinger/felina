"""Loading the centre/spread CSV, and the moment matching it drives."""

import numpy as np
import pytest

from models.center_spread import center_spread, load_targets


def _csv(tmp_path, text):
    path = tmp_path / "moments.csv"
    path.write_text(text)
    return path


def test_loads_the_challenges_own_target_names(tmp_path):
    # The scored dataset names its columns: a cell matches a target column, or the shorthand reported for
    # one. This challenge's shorthand is the column itself, so both routes land on the same name.
    path = _csv(tmp_path, "isoform,mean,std\npEC50,5.0,1.2\n")
    assert load_targets(path) == {"pEC50": (5.0, 1.2)}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("isoform,mean\npEC50,5.0\n", "no column"),
        ("isoform,mean,std\n", "no rows"),
        ("isoform,mean,std\npEC50,high,1.2\n", "must be numbers"),
        ("isoform,mean,std\npEC50,5.0,0\n", "above zero"),
        ("isoform,mean,std\npEC50,5.0,-1\n", "above zero"),
        ("isoform,mean,std\npEC50,nan,1.2\n", "finite"),
        ("isoform,mean,std\npEC50,5.0,1.2\npEC50,4.0,1.0\n", "twice"),
    ],
)
def test_rejects_bad_csv(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_targets(_csv(tmp_path, text))


def test_loads_target_header(tmp_path):
    path = _csv(tmp_path, "target,mean,std\npEC50,5.0,1.2\n")
    assert load_targets(path) == {"pEC50": (5.0, 1.2)}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("target,mean\npEC50,5.0\n", "no column\\(s\\) std"),
        ("mean,std\n5.0,1.2\n", "no column\\(s\\) target"),
        ("target,mean,std\n", "no rows"),
        ("target,mean,std\npEC50,high,1.2\n", "must be numbers"),
        ("target,mean,std\npEC50,5.0,0\n", "above zero"),
        ("target,mean,std\npEC50,5.0,1.2\npEC50,4.0,1.0\n", "twice"),
    ],
)
def test_rejects_bad_csv_with_target_header(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_targets(_csv(tmp_path, text))


def test_missing_file_names_itself(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        load_targets(tmp_path / "absent.csv")


def test_transform_hits_the_target_moments():
    values = np.random.default_rng(0).normal(loc=3.0, scale=0.4, size=200)
    out = center_spread(values, 5.0, 1.2)
    assert out.mean() == pytest.approx(5.0)
    assert out.std() == pytest.approx(1.2)


def test_transform_is_monotone():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    out = center_spread(values, 10.0, 2.0)
    assert np.all(np.diff(out) > 0)  # rank order, so Spearman and Kendall, cannot change


def test_shared_source_scales_parts_together():
    pooled = np.array([0.0, 1.0, 2.0, 3.0])
    first, second = pooled[:2], pooled[2:]
    together = np.concatenate([
        center_spread(first, 5.0, 1.0, source=pooled),
        center_spread(second, 5.0, 1.0, source=pooled),
    ])
    assert together == pytest.approx(center_spread(pooled, 5.0, 1.0))


def test_constant_predictions_are_only_shifted():
    values = np.full(5, 2.0)
    assert center_spread(values, 7.0, 1.5) == pytest.approx(np.full(5, 7.0))
