"""Discovery of the project's dataset definitions.

Every dataset is one module under `data_tools.dataset_defs` exporting a `DatasetSpec` named `SPEC`, holding its
URL, fetch step, derive step, and training heads. This module finds them; it knows nothing about any particular
dataset. To add one, add a module there.
"""

import importlib
import pkgutil
from functools import cache

from data_tools.dataset_spec import DatasetSpec, Source, Step  # noqa: F401  (re-exported for convenience)

Dataset = DatasetSpec  # the spec is the dataset; the old name still works


def _collect(package) -> list[DatasetSpec]:
    """Every spec in a package. A dataset may be a module or a package; a package without a SPEC is a group."""
    specs: list[DatasetSpec] = []
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(info.name)
        spec = getattr(module, "SPEC", None)
        if isinstance(spec, DatasetSpec):
            # A dataset. Whatever else is inside it is its own implementation, not more datasets.
            specs.append(spec)
        elif info.ispkg:
            specs.extend(_collect(module))  # a folder grouping datasets, like metabolism/
        else:
            raise ValueError(f"{info.name} defines no SPEC; every module in dataset_defs must export one")
    return specs


@cache
def registry() -> dict[str, DatasetSpec]:
    """Every known dataset by name, in listing order."""
    from data_tools import dataset_defs

    specs = _collect(dataset_defs)
    specs.sort(key=lambda s: (s.order, s.name))
    datasets: dict[str, DatasetSpec] = {}
    sources: dict[str, str] = {}
    for spec in specs:
        if spec.name in datasets:
            raise ValueError(f"two datasets are both named {spec.name!r}")
        for source in spec.sources:
            if source in sources:
                raise ValueError(f"datasets {sources[source]!r} and {spec.name!r} both define the source {source!r}")
            sources[source] = spec.name
        datasets[spec.name] = spec
    return datasets


def challenge() -> DatasetSpec:
    """The dataset a challenge is scored on.

    `CHALLENGE_DATASET` names it; unset, it is the single dataset declaring `scored=True`. Everything that
    needs the scored target columns, the molecule column, or the blind set asks this rather than holding its
    own copy, so a new challenge is a new scored dataset module and nothing else.
    """
    import os

    datasets = registry()
    wanted = os.environ.get("CHALLENGE_DATASET", "").strip()
    if wanted:
        spec = datasets.get(wanted)
        if spec is None:
            raise ValueError(f"CHALLENGE_DATASET names {wanted!r}, which is not a known dataset")
        if not spec.scored:
            raise ValueError(f"CHALLENGE_DATASET names {wanted!r}, which does not declare scored=True")
        return spec
    scored = [s for s in datasets.values() if s.scored]
    if not scored:
        raise ValueError("no dataset declares scored=True, so there is no challenge to evaluate")
    if len(scored) > 1:
        names = ", ".join(s.name for s in scored)
        raise ValueError(f"several datasets declare scored=True ({names}); set CHALLENGE_DATASET to pick one")
    return scored[0]


def blind_spec() -> DatasetSpec | None:
    """The dataset holding the challenge's blinded molecules, or None if it declares none."""
    spec = challenge()
    if not spec.blind:
        return None
    blind = registry().get(spec.blind)
    if blind is None:
        raise ValueError(f"dataset {spec.name!r} names the blind set {spec.blind!r}, which is not a known dataset")
    return blind


def parse_selector(selector: str) -> tuple[DatasetSpec, list[str]]:
    """Parse a `NAME` or `NAME:TARGET,TARGET` selector into its spec and resolved target columns.

    The same grammar `train-model --datasets` uses. Raises ValueError naming the problem.
    """
    name, _, targets = selector.partition(":")
    name = name.strip()
    spec = registry().get(name)
    if spec is None:
        raise ValueError(f"unknown dataset {name!r}; `data list` shows every known dataset")
    if not spec.scored:
        raise ValueError(f"dataset {name!r} is not a scored dataset, so it has no targets to evaluate")
    wanted = [t.strip() for t in targets.split(",") if t.strip()]
    return spec, spec.resolve_targets(wanted)
