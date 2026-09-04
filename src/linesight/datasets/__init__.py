"""Thin dataset loaders, one per source - path handling and nothing else.

``mvtec.py`` is the correctness anchor (a published benchmark), ``aitex.py`` the
real-fabric study set, ``fabric_stain.py`` the phone-resolution set. No loader
imports the model or the pipeline, so a dataset study can be read without
them. Licences are recorded in ``docs/datasets.md``.
"""
