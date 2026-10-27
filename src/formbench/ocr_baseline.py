from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytesseract

# Share of each side cut from a cell crop so the printed border is not read as ink.
INNER_MARGIN = 0.12
# A cell with less ink than this is empty and never reaches Tesseract.
MIN_INK_RATIO = 0.01
# Ink must be clearly darker than the paper: Otsu alone splits noise in an empty cell.
INK_VS_PAPER = 0.7

# Single-line mode; single-word mode (psm 8) is the fallback because psm 7 often
# returns nothing for a lone digit.
DIGITS = "--psm 7 -c tessedit_char_whitelist=0123456789"
DIGITS_WORD = "--psm 8 -c tessedit_char_whitelist=0123456789"
TEXT = "--psm 7"


def _crop(page: np.ndarray, box: dict[str, int]) -> np.ndarray:
    mx, my = int(box["w"] * INNER_MARGIN), int(box["h"] * INNER_MARGIN)
    return page[box["y"] + my : box["y"] + box["h"] - my, box["x"] + mx : box["x"] + box["w"] - mx]


def ink_mask(crop: np.ndarray, paper: float) -> np.ndarray:
    """Binary mask of ink pixels (255) in a grayscale crop."""
    otsu, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(crop, min(otsu, INK_VS_PAPER * paper), 255, cv2.THRESH_BINARY_INV)
    return mask


def _ocr(mask: np.ndarray, config: str) -> tuple[str, float]:
    # Black text on white with a quiet border is what Tesseract reads best.
    image = cv2.copyMakeBorder(255 - mask, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip()]
    if not words:
        return "", 0.0
    return " ".join(w for w, _ in words), sum(c for _, c in words) / len(words) / 100


def read_form(page: np.ndarray, layout: dict[str, Any]) -> dict[str, Any]:
    """Read every header box and cell of an aligned page.

    Returns {"values": {key: value}, "confidence": {key: 0..1}} with the same keys as the
    extraction schema. Empty cells are null with confidence 1.
    """
    paper = float(np.median(page))
    values: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    boxes = [(k, b, TEXT) for k, b in layout["header"].items()]
    boxes += [(k, b, DIGITS) for k, b in layout["fields"].items()]
    for key, box, config in boxes:
        mask = ink_mask(_crop(page, box), paper)
        if np.count_nonzero(mask) / mask.size < MIN_INK_RATIO:
            values[key], confidence[key] = None, 1.0
            continue
        text, conf = _ocr(mask, config)
        if config == DIGITS and not text:
            text, conf = _ocr(mask, DIGITS_WORD)
        if config == DIGITS:
            values[key] = int(text.replace(" ", "")) if text else None
        else:
            values[key] = text or None
        confidence[key] = round(conf, 3)
    return {"values": values, "confidence": confidence}
