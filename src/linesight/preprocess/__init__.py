"""L3 PREPROCESS - fabric ROI in, tiles out.

``flatfield.py`` divides out the illumination field; ``tiling.py`` cuts the
corrected ROI into overlapping tiles that remember their global coordinates,
and pastes per-tile results back into one frame-sized array.
"""
