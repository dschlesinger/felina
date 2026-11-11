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
