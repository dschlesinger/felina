"""One module per dataset, each defining that dataset alone.

`data_tools.datasets.registry()` imports every module here and collects its `SPEC`. To add a dataset, add a module
exporting a `DatasetSpec` named `SPEC`; nothing outside the new file changes. See `data_tools.dataset_spec` for
what a spec holds.
"""
