"""One module per dataset, each defining that dataset alone.

`data_tools.datasets.registry()` imports everything here and collects each `SPEC`. To add a dataset, add a module
exporting a `DatasetSpec` named `SPEC`; nothing outside it changes. See `data_tools.dataset_spec` for the fields.

Three shapes, by how much code a dataset needs:

  module     one file, for a dataset that is a URL and a head pattern (`train.py`, `tdc_hlm.py`).
  package    a directory whose `__init__.py` exports SPEC, for a dataset that brings its own fetch or reshape
             code: that code lives beside the spec (`pubchem_aid1851/`, `p450_family/`) and is private to it.
             Discovery stops at the SPEC and does not treat the implementation files as datasets.
  group      a directory with no SPEC, which just holds related datasets (`metabolism/`). Discovery recurses.
"""
