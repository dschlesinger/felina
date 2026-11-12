"""Find and download the leaderboards of a Gradio-based challenge Space (e.g. openadmet/cyp-challenge).

The Space URL comes from `--url` or the `CHALLENGE_URL` environment variable / `.env` entry. It can be a Space id
(`openadmet/cyp-challenge`), a `huggingface.co/spaces/...` page, or the `*.hf.space` host. The Space's Gradio config is
read to find every "Download CSV" button, each labelled by the tabs it sits in (e.g. `Final Leaderboard / Regression /
Overall`). Pick which to download interactively, or with `--all` / `--select`. Files go to
`<RESULTS_FOLDER>/leaderboards/<space>/` unless `--out` is given, named `<leaderboard>__<UTC timestamp>.csv`. A new
snapshot is saved only if the CSV differs from the latest one of that leaderboard (`--force` overrides), so reruns build
a longitudinal history without duplicates. `HF_TOKEN` is sent if set (needed for private Spaces).
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from data_tools.paths import RESULTS_DIR

URL_ENV = "CHALLENGE_URL"


@dataclass(frozen=True)
class Leaderboard:
    api_name: str
    path: tuple[str, ...]  # tab labels, outermost first

    @property
    def label(self) -> str:
        return " / ".join(self.path) or self.api_name

    @property
    def stem(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", "__".join(self.path) or self.api_name).strip("_").lower()


def resolve_base_url(url: str) -> str:
    """Turn a Space id, Space page URL or `*.hf.space` URL into the app's base URL."""
    url = url.strip().rstrip("/")
    m = re.match(r"^(?:https?://)?huggingface\.co/spaces/([\w.-]+)/([\w.-]+)", url)
    if m or re.fullmatch(r"[\w.-]+/[\w.-]+", url):
        owner, name = m.groups() if m else url.split("/")
        return "https://" + re.sub(r"[._]", "-", f"{owner}-{name}").lower() + ".hf.space"
    if not url.startswith("http"):
        url = "https://" + url
    return url


def _request(url: str, data: dict | None = None):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(data).encode() if data is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=120)


def find_leaderboards(base: str) -> list[Leaderboard]:
    """List every leaderboard download button of the app, with the tab path it sits under."""
    with _request(urljoin(base + "/", "config")) as r:
        config = json.load(r)
    components = {c["id"]: c for c in config["components"]}
    parents: dict[int, int] = {}

    def walk(node: dict, parent: int | None) -> None:
        if parent is not None:
            parents[node["id"]] = parent
        for child in node.get("children", []):
            walk(child, node["id"])

    walk(config["layout"], None)

    api_by_button = {
        target[0]: dep["api_name"]
        for dep in config["dependencies"]
        for target in dep.get("targets", [])
        if target[1] == "click" and dep.get("api_name")
    }
    found = []
    for cid, component in components.items():
        if component["type"] != "downloadbutton" or cid not in api_by_button:
            continue
        path, node = [], cid
        while node in parents:
            node = parents[node]
            if node in components and components[node]["type"] == "tabitem":
                label = re.sub(r"^[^\w]+", "", components[node]["props"].get("label") or "").strip()
                path.append(label)
        found.append(Leaderboard(api_by_button[cid], tuple(reversed(path))))
    return sorted(found, key=lambda lb: int(re.sub(r"\D", "", lb.api_name) or 0))


def latest_snapshot(out_dir: Path, lb: Leaderboard) -> Path | None:
    """Most recent saved snapshot of this leaderboard (`<stem>__<UTC stamp>.csv`; an un-stamped `<stem>.csv` counts as oldest)."""
    stamped = sorted(out_dir.glob(f"{lb.stem}__*.csv"))
    legacy = out_dir / f"{lb.stem}.csv"
    return stamped[-1] if stamped else legacy if legacy.is_file() else None


def download(base: str, lb: Leaderboard, out_dir: Path, force: bool = False) -> tuple[Path | None, str]:
    """Fetch the leaderboard CSV and save it as `<stem>__<UTC timestamp>.csv`, unless it is identical to the latest snapshot.

    Returns (new file or None, message). A snapshot is kept only when the content differs from the previous one, so the
    folder holds exactly the points in time where the leaderboard changed (`force` keeps it regardless).
    """
    with _request(f"{base}/gradio_api/call/{lb.api_name}", {"data": []}) as r:
        event_id = json.load(r)["event_id"]
    result = None
    with _request(f"{base}/gradio_api/call/{lb.api_name}/{event_id}") as r:
        event = None
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                if event == "complete":
                    result = json.loads(line[5:])
                elif event == "error":
                    raise RuntimeError(f"{lb.label}: {line[5:].strip()}")
    if not result or not isinstance(result[0], dict) or not result[0].get("url"):
        raise RuntimeError(f"{lb.label}: no file returned ({result!r})")
    with _request(result[0]["url"]) as r:
        data = r.read().replace(b"\r\n", b"\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = latest_snapshot(out_dir, lb)
    if previous is not None and not force:
        old = previous.read_bytes().replace(b"\r\n", b"\n")
        if old == data:
            return None, f"unchanged since {previous.name}"
        delta = data.count(b"\n") - old.count(b"\n")
        note = f"changed vs {previous.name} ({delta:+d} rows)"
    else:
        note = "first snapshot" if previous is None else "forced"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / f"{lb.stem}__{stamp}.csv"
    dest.write_bytes(data)
    return dest, note


def parse_selection(text: str, boards: list[Leaderboard]) -> list[Leaderboard]:
    """Pick leaderboards by 1-based numbers/ranges (`1,3-5`), `all`, or a case-insensitive substring of the label."""
    text = text.strip()
    if text.lower() == "all":
        return boards
    chosen: list[Leaderboard] = []
    for part in (p.strip() for p in text.split(",") if p.strip()):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            if not 1 <= lo <= hi <= len(boards):
                raise SystemExit(f"Selection {part!r} out of range 1-{len(boards)}")
            matches = boards[lo - 1 : hi]
        else:
            matches = [b for b in boards if part.lower() in b.label.lower()]
            if not matches:
                raise SystemExit(f"No leaderboard matches {part!r}")
        chosen += [b for b in matches if b not in chosen]
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get(URL_ENV), help=f"challenge Space (default: ${URL_ENV})")
    parser.add_argument("--list", action="store_true", help="only list the available leaderboards")
    parser.add_argument("--all", action="store_true", help="download every leaderboard without asking")
    parser.add_argument("--select", help="numbers/ranges (1,3-5) or label substrings (final,interim); comma-separated")
    parser.add_argument("--force", action="store_true", help="save a snapshot even if nothing changed since the last one")
    parser.add_argument("--out", type=Path, help="output folder (default: <RESULTS_FOLDER>/leaderboards/<space>)")
    args = parser.parse_args(argv)
    if not args.url:
        parser.error(f"give --url or set {URL_ENV} (environment or .env)")

    base = resolve_base_url(args.url)
    boards = find_leaderboards(base)
    if not boards:
        raise SystemExit(f"No downloadable leaderboards found at {base}")
    for i, lb in enumerate(boards, 1):
        print(f"{i:3d}  {lb.label}")
    if args.list:
        return

    if args.all:
        chosen = boards
    elif args.select:
        chosen = parse_selection(args.select, boards)
    elif sys.stdin.isatty():
        chosen = parse_selection(input("Download which? (numbers, ranges, text, or 'all'): "), boards)
    else:
        raise SystemExit("Not a terminal: pass --all or --select")

    out_dir = args.out or RESULTS_DIR / "leaderboards" / re.sub(r"^https?://|\.hf\.space$", "", base)
    for lb in chosen:
        try:
            dest, note = download(base, lb, out_dir, args.force)
            print(f"{lb.label}: {note}" + (f" -> {dest}" if dest else ""))
        except Exception as exc:
            print(f"{lb.label}: FAILED ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
