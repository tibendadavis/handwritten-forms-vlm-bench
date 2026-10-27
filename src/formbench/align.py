from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class MarkersNotFound(ValueError):
    """The four corner markers could not be located in the photo."""


def load_gray(path: Path) -> np.ndarray:
    # imread applies EXIF orientation, so phone photos come in upright.
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image {path}")
    return image


def _candidates(binary: np.ndarray) -> list[tuple[float, float]]:
    """Centres of solid, roughly square blobs of marker size."""
    side = min(binary.shape)
    min_area, max_area = (0.01 * side) ** 2, (0.12 * side) ** 2
    # RETR_LIST, not RETR_EXTERNAL: a dark table around the page would otherwise
    # be the only outer contour and hide the markers inside it.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        quad = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(contour)
        if len(quad) != 4 or not 0.5 <= w / h <= 2:
            continue
        # Solid ink inside, which rules out the empty table cells.
        if np.count_nonzero(binary[y : y + h, x : x + w]) / (w * h) < 0.75:
            continue
        m = cv2.moments(contour)
        centres.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return centres


def _corners(centres: list[tuple[float, float]], shape: tuple[int, ...]) -> np.ndarray | None:
    if len(centres) < 4:
        return None
    pts = np.array(centres)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    picks = [s.argmin(), d.argmax(), s.argmax(), d.argmin()]  # TL, TR, BR, BL
    if len(set(picks)) < 4:
        return None
    quad = pts[picks].astype(np.float32)
    if cv2.contourArea(quad) < 0.1 * shape[0] * shape[1]:
        return None
    return quad


def find_markers(gray: np.ndarray) -> np.ndarray:
    """Marker centres in the photo, ordered TL, TR, BR, BL."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Fallback for uneven light; the block is larger than a marker so markers stay solid.
    block = (min(gray.shape) // 8) | 1
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15
    )
    found = 0
    for binary in (otsu, adaptive):
        centres = _candidates(binary)
        found = max(found, len(centres))
        quad = _corners(centres, gray.shape)
        if quad is not None:
            return quad
    raise MarkersNotFound(
        f"Found {found} marker-like squares but could not locate all four corner markers. "
        "Keep the four black corner squares in the frame, lay the page flat and upright, "
        "and avoid shadows over the corners."
    )


def marker_centres(layout: dict[str, Any]) -> np.ndarray:
    return np.array(
        [(m["x"] + m["w"] / 2, m["y"] + m["h"] / 2) for m in layout["markers"]], dtype=np.float32
    )


def align(image: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    """Warp a photo of the form onto the template's page, in grayscale page pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    matrix = cv2.getPerspectiveTransform(find_markers(gray), marker_centres(layout))
    size = (layout["page"]["width"], layout["page"]["height"])
    return cv2.warpPerspective(gray, matrix, size, flags=cv2.INTER_LINEAR, borderValue=255)
