import argparse
import os
import subprocess
import sys

import pytest

from models.run_log import changed_flags, list_runs, main, run_log


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--plots", action="store_true")
    return parser


def test_changed_flags_only_non_defaults():
    parser = _parser()
    args = parser.parse_args(["--models", "xgb", "rf", "--plots"])
    assert changed_flags(args, parser) == {"models": ["xgb", "rf"], "plots": True}


def test_run_log_names_ids_and_captures_output(tmp_path, capfd):
    with run_log("eval", {"models": ["xgb", "rf"], "plots": True}, runs_dir=tmp_path):
        os.write(1, b"from python\n")  # fd level: pytest swaps sys.stdout for its own object
        subprocess.run(["echo", "from child"], check=True)
        os.write(2, b"to stderr\n")
    with run_log("train", {"model": "chemprop", "seed": 7}, runs_dir=tmp_path) as log:
        log.code = 1

    first = tmp_path / "0001__eval__models=xgb+rf_plots.log"
    assert first.is_file()
    text = first.read_text()
    assert "from python" in text and "from child" in text and "to stderr" in text
    assert "status=ok" in text.splitlines()[-1]
    assert "from python" in capfd.readouterr().out  # still shown to the user

    runs = list_runs(tmp_path)
    assert [(r.id, r.kind) for r in runs] == [(2, "train"), (1, "eval")]
    assert runs[0].status == "FAILED(1)"


def test_failure_is_recorded(tmp_path):
    with pytest.raises(SystemExit):
        with run_log("eval", {}, runs_dir=tmp_path):
            sys.exit(3)
    assert list_runs(tmp_path)[0].status == "FAILED(3)"


def test_list_runs_filters(tmp_path, capsys):
    for kind in ("train", "eval", "train"):
        with run_log(kind, {}, runs_dir=tmp_path):
            pass
    main(["--dir", str(tmp_path), "--train-model"])
    out = capsys.readouterr().out
    assert "train" in out and "eval" not in out and "2 runs" in out
    main(["--dir", str(tmp_path), "--eval"])
    assert "1 run in" in capsys.readouterr().out
