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

    first = tmp_path / "0001__eval__models=xgb+rf_plots" / "run.log"
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



def test_print_split_routes_terminal_and_log(tmp_path):
    script = (
        "import sys; from pathlib import Path; from models.run_log import print_split, run_log\n"
        "with run_log('eval', {}, runs_dir=Path(sys.argv[1])):\n"
        "    print('shared', flush=True)\n"
        "    print_split('short\\u2026\\n', 'full name\\n')\n"
        "    print('after', flush=True)\n"
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    shown = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env, check=True).stdout
    assert shown.splitlines() == ["shared", "short\u2026", "after"]
    body = next(tmp_path.glob("*/run.log")).read_text().split("# ---\n", 1)[1]
    assert body.splitlines()[:3] == ["shared", "full name", "after"]


def test_run_gets_its_own_folder(tmp_path):
    with run_log("eval", {"models": ["ridge"]}, runs_dir=tmp_path) as log:
        assert log.dir == tmp_path / "0001__eval__models=ridge"
        assert log.dir.is_dir()
        (log.dir / "results.md").write_text("# results\n")
    run = list_runs(tmp_path)[0]
    assert run.path == tmp_path / "0001__eval__models=ridge" / "run.log"
    assert (run.path.parent / "results.md").is_file()


def test_migrate_moves_flat_logs_and_keeps_ids(tmp_path):
    from models.run_log import migrate_flat_logs

    flat = tmp_path / "0001__eval__models=ridge.log"
    flat.write_text("# run: 0001\n# ---\nold output\n# finished: x status=ok duration=3s\n")
    assert [(r.id, r.kind) for r in list_runs(tmp_path)] == [(1, "eval")]  # read before migrating

    moved = migrate_flat_logs(tmp_path)
    assert moved == [(flat, tmp_path / "0001__eval__models=ridge" / "run.log")]
    assert not flat.exists()
    run = list_runs(tmp_path)[0]
    assert (run.id, run.status, run.flags) == (1, "ok", "models=ridge")

    assert migrate_flat_logs(tmp_path) == []  # nothing left to move
    with run_log("eval", {}, runs_dir=tmp_path) as log:  # the migrated id is still taken
        assert log.id == 2


def test_clean_interrupt_reports_one_line(tmp_path):
    script = (
        "import sys; from pathlib import Path\n"
        "from models.run_log import clean_interrupt, run_log\n"
        "with clean_interrupt('evaluate-models'):\n"
        "    with run_log('eval', {}, runs_dir=Path(sys.argv[1])):\n"
        "        raise KeyboardInterrupt\n"
    )
    env = {**os.environ, "PYTHONPATH": "src"}
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env)
    assert done.returncode == 130
    assert done.stderr.strip() == "evaluate-models interrupted (Ctrl-C)"
    assert "Traceback" not in done.stderr
    assert list_runs(tmp_path)[0].status == "interrupted"  # the log still records it
