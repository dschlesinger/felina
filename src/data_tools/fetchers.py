"""Fetch steps shared by several dataset definitions.

A step takes the data directory and writes under it. Anything used by exactly one dataset belongs in that
dataset's module instead; this file is only for what more than one of them needs.
"""

import sys
from pathlib import Path

from data_tools.dataset_spec import Step


def download_csv(url: str, local_path: Path) -> None:
    """Read a CSV (local, http, or `hf://...`) and write it to local_path, replacing whatever is there.

    Writes to a temporary file first, so an interrupted download leaves the previous copy intact.
    """
    import polars as pl

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...", file=sys.stderr)
    frame = pl.read_csv(url)
    partial = local_path.with_name(local_path.name + ".part")
    frame.write_csv(partial)
    partial.replace(local_path)
    print(f"Saved {len(frame)} rows -> {local_path}", file=sys.stderr)


def csv_from(url: str, relative: str) -> Step:
    """A fetch step that downloads `url` to `relative` under the data directory."""

    def fetch(data_dir: Path) -> None:
        download_csv(url, data_dir / relative)

    return fetch
