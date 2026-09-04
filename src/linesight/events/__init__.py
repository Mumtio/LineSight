"""L6 EVENT ASSEMBLY - detections per frame, events per roll.

``assemble.py`` turns one frame's score maps into ``Detection`` objects in
millimetres; ``track.py`` fuses detections across frames into ``Event`` objects
spanning the roll. The split is deliberate: a frame can be assembled, and
tested, with no cross-frame state at all.
"""
