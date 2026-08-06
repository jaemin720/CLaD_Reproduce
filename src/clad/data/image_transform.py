"""Named RGB transforms shared by offline data and online evaluation."""

from __future__ import annotations

import numpy as np

IMAGE_TRANSFORMS = ("none", "flip_vertical", "rotate_180")


def transform_rgb_image(image: np.ndarray, transform: str) -> np.ndarray:
    """Apply a spatial transform to an HWC RGB image or an NHWC image batch."""

    if transform not in IMAGE_TRANSFORMS:
        raise ValueError(
            f"Unsupported image transform {transform!r}; expected one of {IMAGE_TRANSFORMS}"
        )
    array = np.asarray(image)
    if array.ndim not in {3, 4} or array.shape[-1] != 3:
        raise ValueError(
            "RGB image must have shape [H,W,3] or [N,H,W,3], "
            f"got {array.shape}"
        )
    if transform == "flip_vertical":
        array = np.flip(array, axis=-3)
    elif transform == "rotate_180":
        array = np.flip(array, axis=(-3, -2))
    return np.ascontiguousarray(array)
