"""Per-run logs in `<RESULTS_FOLDER>/runs`, and the `list-runs` CLI that reads them back.

Every `train-model` and `evaluate-models` run writes one log, named `<id>__<kind>__<flags>.log`:

    0007__train__model=unimol_datasets=challenge_seed=43.log
    0008__eval__models=xgb+rf_split=butina.log

The id counts up from the highest one in the folder, `kind` is `train` or `eval`, and the flags are the ones that
differ from their default (`key=value`, `+` between list items, a bare `key` for an on-switch). The log itself starts
with a header (full command, git commit, start time) and ends with a `# finished:` line holding status and duration.
"""

import argparse
import codecs
import contextlib
import fcntl
import os
import pty
import re
import shlex
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data_tools.paths import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "runs"
KINDS = ("train", "eval")
_NAME = re.compile(r"^(\d+)__(train|eval)(?:__(.*))?\.log$")
_MAX_FLAGS = 150  # characters of flags kept in a filename; the full command is in the log header
_END_OF_HEADER = "# ---"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[_PX^][^\x1b]*\x1b\\")


def changed_flags(args: argparse.Namespace, parser: argparse.ArgumentParser, skip: set[str] | frozenset[str] = frozenset()) -> dict[str, object]:
    """The flags whose parsed value differs from the parser's default, keyed by flag name (dashes, no leading --)."""
    return {
        dest.replace("_", "-"): value
        for dest, value in sorted(vars(args).items())
        if dest not in skip and not dest.startswith("_") and value != parser.get_default(dest)
    }


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "-", text)


def _value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "+".join(_value(item) for item in value)
    if isinstance(value, Path):
        return _clean(value.name or str(value))
    return _clean(str(value))


def _filename(run_id: int, kind: str, flags: Mapping[str, object]) -> str:
    body = "_".join(key if value is True else f"{key}={_value(value)}" for key, value in flags.items())
    if len(body) > _MAX_FLAGS:
        body = body[:_MAX_FLAGS].rsplit("_", 1)[0] + "_etc"
    return f"{run_id:04d}__{kind}" + (f"__{body}" if body else "") + ".log"


def _open_log(kind: str, flags: Mapping[str, object], runs_dir: Path):
    """Create the next log file; the lock keeps parallel jobs from taking the same id."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ids = [int(m[1]) for p in runs_dir.glob("*.log") if (m := _NAME.match(p.name))]
        run_id = max(ids, default=0) + 1
        path = runs_dir / _filename(run_id, kind, flags)
        return run_id, path, open(path, "x", encoding="utf-8")


def _git_state() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], capture_output=True, timeout=5).returncode != 0
        return (rev.stdout.strip() or "unknown") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class RunLog:
    """The open log of one run; set `code` to a nonzero exit code to mark the run failed."""

    def __init__(self, run_id: int, path: Path, handle) -> None:
        self.id = run_id
        self.path = path
        self.code: int | None = None
        self._handle = handle
        self._lock = threading.Lock()

    def line(self, text: str) -> None:
        with self._lock:
            self._handle.write(text + "\n")
            self._handle.flush()


def _clean_line(line: str) -> str:
    """Drop colour/image escape codes and keep only the last state of a carriage-return progress bar."""
    return _ANSI.sub("", line).rstrip("\r").rsplit("\r", 1)[-1]


class _Tee:
    """Copy everything written to a file descriptor (this process and its children) into the log, still showing it."""

    def __init__(self, fd: int, log: RunLog) -> None:
        self.fd = fd
        self.log = log
        self.saved = os.dup(fd)
        if os.isatty(fd):  # a pty keeps colours, progress bars and inline images working for the user
            self.read_fd, self.write_fd = pty.openpty()
            with contextlib.suppress(OSError):
                fcntl.ioctl(self.write_fd, termios.TIOCSWINSZ, fcntl.ioctl(fd, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0)))
        else:
            self.read_fd, self.write_fd = os.pipe()
        os.dup2(self.write_fd, fd)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _show(self, chunk: bytes) -> None:
        with contextlib.suppress(OSError):
            while chunk:
                chunk = chunk[os.write(self.saved, chunk):]

    def _pump(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        while True:
            try:
                chunk = os.read(self.read_fd, 65536)
            except OSError:  # the pty reports EIO once every writer has closed
                break
            if not chunk:
                break
            self._show(chunk)
            *lines, pending = (pending + decoder.decode(chunk)).split("\n")
            for line in lines:
                self.log.line(_clean_line(line))
        pending += decoder.decode(b"", final=True)
        if pending:
            self.log.line(_clean_line(pending))
        os.close(self.read_fd)

    def close(self) -> None:
        os.dup2(self.saved, self.fd)  # restores the original, dropping this process's copy of the pipe
        os.close(self.write_fd)
        self.thread.join(timeout=5)
        os.close(self.saved)


@contextlib.contextmanager
def run_log(kind: str, flags: Mapping[str, object], header: Mapping[str, str] | None = None, runs_dir: Path | None = None) -> Iterator[RunLog]:
    """Log stdout and stderr of the enclosed block to the next `<id>__<kind>__<flags>.log` while still printing them."""
    run_id, path, handle = _open_log(kind, flags, runs_dir or RUNS_DIR)
    log = RunLog(run_id, path, handle)
    started = time.time()
    lines = {
        "run": f"{run_id:04d}",
        "kind": kind,
        "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "git": _git_state(),
        "host": socket.gethostname(),
        **(header or {}),
    }
    for key, value in lines.items():
        log.line(f"# {key}: {value}")
    log.line(_END_OF_HEADER)
    sys.stdout.flush()
    sys.stderr.flush()
    tees = [_Tee(1, log), _Tee(2, log)]
    status = "ok"
    try:
        yield log
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status = f"FAILED({exc.code})"
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "FAILED(error)"
        log.line(traceback.format_exc().rstrip())
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for tee in tees:
            tee.close()
        if status == "ok" and log.code not in (None, 0):
            status = f"FAILED({log.code})"
        log.line(f"# finished: {datetime.now().isoformat(timespec='seconds')} status={status} duration={round(time.time() - started)}s")
        handle.close()


@dataclass
class RunInfo:
    id: int
    kind: str
    flags: str
    path: Path
    started: str
    status: str
    duration: str


def _duration(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m" if minutes else f"{secs}s"


def _display_flags(raw: str | None) -> str:
    return " ".join(part.replace("+", ",") for part in (raw or "").split("_") if part)


def read_run(path: Path) -> RunInfo | None:
    """Parse a log's filename, header and last line; None if the file is not a run log."""
    match = _NAME.match(path.name)
    if not match:
        return None
    started = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    status, duration = "unfinished", ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for _, line in zip(range(30), handle):
            if line.startswith("# started: "):
                started = line.split(": ", 1)[1].strip()
            if line.startswith(_END_OF_HEADER):
                break
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 2048))
        done = [line for line in handle.read().splitlines() if line.startswith("# finished: ")]
    if done:
        fields = dict(part.split("=", 1) for part in done[-1].split() if "=" in part)
        status = fields.get("status", "?")
        duration = _duration(int(fields["duration"].rstrip("s"))) if "duration" in fields else ""
    return RunInfo(int(match[1]), match[2], _display_flags(match[3]), path, started.replace("T", " ")[:16], status, duration)


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Every run log in the folder, newest first."""
    runs = [info for p in runs_dir.glob("*.log") if (info := read_run(p))]
    return sorted(runs, key=lambda r: r.id, reverse=True)


def _table(runs: list[RunInfo]) -> str:
    rows = [("ID", "WHEN", "KIND", "STATUS", "TIME", "FLAGS")]
    rows += [(f"{r.id:04d}", r.started, r.kind, r.status, r.duration, r.flags) for r in runs]
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    return "\n".join(("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[5]).rstrip() for row in rows)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the list-runs CLI."""
    parser = argparse.ArgumentParser(prog="list-runs", description="List previous train-model and evaluate-models runs, newest first.")
    parser.add_argument("run_id", nargs="?", type=int, help="Show one run in full: command, commit, log path and the end of its output")
    parser.add_argument("--train-model", action="store_true", help="Only train-model runs")
    parser.add_argument("--eval", action="store_true", help="Only evaluate-models runs")
    parser.add_argument("-n", type=int, default=None, metavar="N", help="Only the newest N runs")
    parser.add_argument("--dir", type=Path, default=RUNS_DIR, help="Folder with the run logs (default: <RESULTS_FOLDER>/runs)")
    parser.add_argument("--tail", type=int, default=20, help="Lines of output shown for a single run (default: 20)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    runs = list_runs(args.dir)
    if args.run_id is not None:
        run = next((r for r in runs if r.id == args.run_id), None)
        if run is None:
            sys.exit(f"No run {args.run_id:04d} in {args.dir}")
        text = run.path.read_text(encoding="utf-8", errors="replace").splitlines()
        end = text.index(_END_OF_HEADER) if _END_OF_HEADER in text else 0
        print("\n".join(text[:end]))
        print(f"# log: {run.path}\n# status: {run.status} {run.duration}".rstrip())
        print(_END_OF_HEADER)
        print("\n".join(text[end + 1:][-args.tail:]))
        return

    wanted = {kind for kind, flag in (("train", args.train_model), ("eval", args.eval)) if flag} or set(KINDS)
    runs = [r for r in runs if r.kind in wanted]
    if args.n is not None:
        runs = runs[: args.n]
    if not runs:
        print(f"No runs in {args.dir}")
        return
    print(_table(runs))
    print(f"\n{len(runs)} run{'s' if len(runs) != 1 else ''} in {args.dir}")


if __name__ == "__main__":
    main()
