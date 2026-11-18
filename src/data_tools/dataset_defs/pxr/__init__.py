"""The OpenADMET PXR Induction Blind Challenge: the splits it ships, and the Emax data alongside them.

A group, not a dataset: it exports no SPEC, so discovery recurses into it and collects the modules beside this
one. They share `_reshape.py`, which discovery skips because its name starts with an underscore, holding the
column mapping and the derive steps they all need.
"""
