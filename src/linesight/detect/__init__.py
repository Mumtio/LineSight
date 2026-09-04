"""L4 DETECTION - the seam: a tile in, an unbounded score map out.

``base.py`` holds the protocol every scorer implements. ``backbone.py`` is the
frozen feature extractor, ``patchcore.py`` the shipped memory-bank scorer, and
``stub.py`` a numpy-only scorer used to exercise the other layers without a
model. Which one runs is ``config.detect.scorer``.
"""
