# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    h, w = img.shape

    # 1. Translated image (shift right and down by 20 pixels)
    tx, ty = 20, 20
    translated = np.zeros_like(img)
    translated[ty:, tx:] = img[:h-ty, :w-tx]

    # 2. Rotated image (90 degrees clockwise)
    rotated = np.transpose(img[::-1, :])

    # 3. Horizontally stretched image (scale width by 1.5)
    new_w = int(w * 1.5)
    stretched = np.zeros((h, new_w), dtype=img.dtype)
    x_idx = (np.linspace(0, w-1, new_w)).astype(int)
    stretched[:, :] = img[:, x_idx]

    # 4. Horizontally mirrored image (flip along vertical axis)
    mirrored = img[:, ::-1]

    # 5. Barrel distorted image (radial distortion)
    distorted = np.zeros_like(img)
    cx, cy = w // 2, h // 2
    y, x = np.indices((h, w))
    x_norm = (x - cx) / cx
    y_norm = (y - cy) / cy
    r = np.sqrt(x_norm**2 + y_norm**2)

    # fator de distorção (quanto maior k, mais forte o efeito barril)
    k = 0.3
    r_distorted = r * (1 + k * r**2)

    x_distorted = (x_norm / (r + 1e-8)) * r_distorted * cx + cx
    y_distorted = (y_norm / (r + 1e-8)) * r_distorted * cy + cy

    x_distorted = np.clip(x_distorted.round().astype(int), 0, w-1)
    y_distorted = np.clip(y_distorted.round().astype(int), 0, h-1)

    distorted[y, x] = img[y_distorted, x_distorted]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }