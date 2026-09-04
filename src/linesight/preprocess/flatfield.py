"""Flat-field correction - the first thing to check when the heatmap is noise.

A camera over a bench rig produces a bright centre and dark corners. A memory
bank fitted on unevenly-lit fabric spends its capacity modelling the lamp
rather than the weave, and then every corner scores as anomalous. Uneven
illumination is the most common single cause of a uniformly noisy heatmap, so
this runs before tiling on every path - fit, calibrate and inspect alike (see
``Pipeline.prepare``).

Two modes:
  * **self** - estimate the illumination field from the frame itself with a
    wide Gaussian blur and divide it out. No calibration shot needed, and it
    works well on texture, which is why it is the default.
  * **reference** - divide by a stored blank-field image of the empty rig.
    More accurate where such a capture exists; ``save_reference`` /
    ``load_reference`` persist it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["estimate_illumination", "flat_field", "load_reference", "save_reference"]

#: Below this the division amplifies sensor noise into fake defects rather than
#: correcting anything.
_MIN_ILLUMINATION = 1.0


def estimate_illumination(image: np.ndarray, sigma_px: float = 101.0) -> np.ndarray:
    """Low-frequency illumination estimate: a heavily blurred copy of the image.

    ``sigma_px`` must be large relative to the defects (a 1 mm slub at 0.1 mm/px
    is 10 px) and small relative to the lighting gradient. Too small and the
    correction eats the very defects it is meant to expose.

    Returns:
        float32, same shape as the input, clipped strictly positive so the
        division downstream cannot blow up in a genuinely black corner.
    """
    array = np.asarray(image).astype(np.float32)
    # OpenCV wants an odd kernel; derive it from sigma rather than asking for
    # both and letting them disagree.
    ksize = max(3, int(sigma_px) | 1)
    blurred = cv2.GaussianBlur(array, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)
    return np.maximum(blurred, _MIN_ILLUMINATION)


def flat_field(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    sigma_px: float = 101.0,
    preserve_mean: bool = True,
) -> np.ndarray:
    """Divide out the illumination field.

    Args:
        reference: a blank-field image; if None, estimated from ``image`` itself.
        preserve_mean: rescale so the output keeps the input's mean intensity,
            which keeps the result a valid uint8 image the backbone can eat.

    Returns:
        uint8, same shape as the input.

    Raises:
        ValueError: if a reference is given whose shape does not match.
    """
    array = np.asarray(image)
    original = array.astype(np.float32)

    if reference is None:
        illumination = estimate_illumination(array, sigma_px)
    else:
        ref = np.asarray(reference)
        if ref.shape != array.shape:
            raise ValueError(
                f"reference shape {ref.shape} does not match frame {array.shape}"
            )
        illumination = np.maximum(ref.astype(np.float32), _MIN_ILLUMINATION)

    corrected = original / illumination
    if preserve_mean:
        corrected *= float(original.mean()) / max(float(corrected.mean()), 1e-6)

    return np.clip(corrected, 0, 255).astype(np.uint8)


def load_reference(path: str | Path) -> np.ndarray:
    """Load a stored blank-field capture.

    Raises:
        OSError: if the file cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"could not decode blank-field reference {path}")
    return image


def save_reference(image: np.ndarray, path: str | Path) -> None:
    """Persist a blank-field capture, averaged over several frames by the caller.

    Averaging is the caller's job because the right number of frames depends on
    the rig's noise, and baking a count in here would hide that choice.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), np.asarray(image)):
        raise OSError(f"could not write blank-field reference to {out}")
