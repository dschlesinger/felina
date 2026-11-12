"""Default data, feature-cache and results folders, configurable through a `.env` file.

`DATA_FOLDER`, `FEATURE_FOLDER` and `RESULTS_FOLDER` are read from the environment, after loading a `.env` file in the
working directory (variables already set in the environment win over the file). Unset, they default to `data`,
`<DATA_FOLDER>/features` and `results`. A command-line flag still overrides all of this.
"""

import os
from pathlib import Path


def load_env_file(path: Path = Path(".env")) -> None:
    """Set variables from a KEY=VALUE file into os.environ without overriding ones that are already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file()

DATA_DIR = Path(os.environ.get("DATA_FOLDER") or "data")
FEATURE_DIR = Path(os.environ.get("FEATURE_FOLDER") or DATA_DIR / "features")
RESULTS_DIR = Path(os.environ.get("RESULTS_FOLDER") or "results")
