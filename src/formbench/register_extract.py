"""Reading register photos with a VLM: the whole page, or aligned strips of rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from formbench.register_schema import (
    REGISTER_PROMPT_VERSION,
    ROW_FIELDS,
    cohort_prompt,
    cohort_schema,
    cover_prompt,
    cover_schema,
    page_prompt,
    page_schema,
    rows_prompt,
    rows_schema,
)

# Columns the model needs. B (first visit), E (name), F (guardian/phone), H (village) and the
# Vitamin A / deworming / AEFI columns are cut out of tiles, so they never reach the model.
TILE_COLUMNS = ["A", "C", "D", "G", *"IJKLMNOPQRSTUVWXY"]
ROWS_PER_TILE = 4
FULL_MAX_SIDE = 3072
TILE_MAX_SIDE = 2400
ALIGN_WIDTH = 1600  # features are matched on downscaled copies; the warp is full resolution
MIN_INLIERS = 60
MIN_GRID_OVERLAP = 0.85


class AlignmentFailed(ValueError):
    """The photo could not be matched to the blank register page."""


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def align_to_template(photo: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Warp a register photo onto the blank template's pixels (no markers needed).

    ORB features on the printed header text and lines, ratio-tested matches and a RANSAC
    homography; too few consistent matches raises AlignmentFailed rather than a bad warp.
    """
    photo = _gray(photo)
    sp, st = ALIGN_WIDTH / photo.shape[1], ALIGN_WIDTH / template.shape[1]
    small_p = cv2.resize(photo, None, fx=sp, fy=sp, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(template, None, fx=st, fy=st, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=8000)
    kp_p, des_p = orb.detectAndCompute(small_p, None)
    kp_t, des_t = orb.detectAndCompute(small_t, None)
    if des_p is None or des_t is None:
        raise AlignmentFailed("no features found in the photo")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_p, des_t, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        raise AlignmentFailed(f"only {len(good)} feature matches")
    src = np.float32([kp_p[m.queryIdx].pt for m in good])
    dst = np.float32([kp_t[m.trainIdx].pt for m in good])
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if homography is None or int(inliers.sum()) < MIN_INLIERS:
        raise AlignmentFailed("no consistent page position found")
    homography = _refine(small_p, small_t, homography)
    scale_p, scale_t = np.diag([sp, sp, 1.0]), np.diag([1 / st, 1 / st, 1.0])
    full = scale_t @ homography @ scale_p
    h, w = template.shape
    aligned = cv2.warpPerspective(photo, full, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    score = grid_overlap(aligned, template)
    if score < MIN_GRID_OVERLAP:
        raise AlignmentFailed(f"printed grid does not line up ({score:.0%} overlap)")
    return aligned


def _refine(small_p: np.ndarray, small_t: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Polish the feature homography with ECC over the whole image.

    Matched features sit mostly in the printed header (the body is handwriting), so the
    first estimate drifts towards the bottom; ECC pulls the full printed grid into place.
    If ECC does not converge, the feature estimate is kept and the overlap check decides.
    """
    template = cv2.GaussianBlur(small_t, (0, 0), 1.5).astype(np.float32) / 255
    photo = cv2.GaussianBlur(small_p, (0, 0), 1.5).astype(np.float32) / 255
    warp = np.linalg.inv(homography).astype(np.float32)  # ECC maps template -> photo
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            template, photo, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error:
        return homography
    return np.linalg.inv(warp.astype(np.float64))


def _grid_lines(gray: np.ndarray) -> np.ndarray:
    ink = (gray < 160).astype(np.uint8) * 255
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))
    return cv2.bitwise_or(h, v)


def grid_overlap(aligned: np.ndarray, template: np.ndarray) -> float:
    """Share of the template's printed table lines that land on dark pixels after the warp.

    Feature matching can lock onto the wrong repeat of a regular grid; this catches it.
    """
    lines = _grid_lines(template) > 0
    dark = cv2.dilate((aligned < 170).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float(dark[lines].mean()) if lines.any() else 0.0


def _columns(page: np.ndarray, layout: dict[str, Any], y0: int, y1: int) -> np.ndarray:
    return np.hstack([page[y0:y1, slice(*layout["columns"][c])] for c in TILE_COLUMNS])


def tiles(page: np.ndarray, layout: dict[str, Any]) -> list[tuple[Image.Image, list[int]]]:
    """Strips of ROWS_PER_TILE rows, each under the column header, with only TILE_COLUMNS."""
    header = _columns(page, layout, layout["header"]["y0"], layout["header"]["y1"])
    rows = layout["rows"]
    out = []
    for start in range(0, len(rows), ROWS_PER_TILE):
        chunk = rows[start : start + ROWS_PER_TILE]
        body = _columns(page, layout, chunk[0]["y0"], chunk[-1]["y1"])
        out.append(
            (Image.fromarray(np.vstack([header, body])), list(range(start, start + len(chunk))))
        )
    return out


def cohort_crop(page: np.ndarray, layout: dict[str, Any]) -> Image.Image:
    x0, y0 = layout["month_box"][0] - 200, layout["month_box"][1]
    x1, y1 = layout["year_box"][2], layout["year_box"][3]
    return Image.fromarray(page[y0:y1, x0:x1])


def clean_rows(rows: Any) -> list[dict[str, Any]]:
    """Keep only schema keys, as strings or None; drop rows with nothing written."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        clean = {
            k: (str(row[k]).strip() or None) if row.get(k) not in (None, "") else None
            for k in ROW_FIELDS
        }
        if any(clean.values()):
            out.append(clean)
    return out


def _call(image, schema, prompt, model, think, max_side, name):
    from formbench.vlm import complete_json, interpret

    raw = complete_json(
        image, schema, prompt, model, think=think, max_side=max_side, name=name,
        metadata={"prompt_version": REGISTER_PROMPT_VERSION},
    )  # fmt: skip
    return interpret(raw, schema)


def read_page(
    photo: Path,
    model: str,
    *,
    mode: str,
    template: np.ndarray,
    layout: dict[str, Any],
    think: bool = False,
) -> dict[str, Any]:
    """Transcribe one cohort page. Returns the reading plus validity, raw replies and notes.

    `aligned` (the photo warped onto the template) is returned when alignment worked, for
    review crops; in tiles mode a failed alignment falls back to the full page with a note.
    """
    image = cv2.imread(str(photo))
    if image is None:
        raise ValueError(f"Cannot read image {photo}")
    notes: list[str] = []
    try:
        aligned = align_to_template(image, template)
    except AlignmentFailed as error:
        aligned = None
        notes.append(f"alignment failed: {error}")
        if mode == "tiles":
            notes.append("fell back to full-page reading")
            mode = "full"

    if mode == "full":
        result = _call(
            photo, page_schema(), page_prompt(), model, think, FULL_MAX_SIDE, "register-page"
        )
        values = result["values"]
        reading = {
            "cohort_month": values.get("cohort_month"),
            "cohort_year": values.get("cohort_year"),
            "rows": clean_rows(values.get("rows")),
        }
        return {"reading": reading, "mode": mode, "json_valid": result["json_valid"],
                "errors": result["errors"], "raw": [result["raw"]], "notes": notes,
                "aligned": aligned}  # fmt: skip

    cohort = _call(cohort_crop(aligned, layout), cohort_schema(), cohort_prompt(), model, think, TILE_MAX_SIDE, "register-cohort")  # fmt: skip
    results = [cohort]
    rows: list[dict[str, Any]] = []
    for strip, _ in tiles(aligned, layout):
        result = _call(strip, rows_schema(), rows_prompt(), model, think, TILE_MAX_SIDE, "register-rows")  # fmt: skip
        results.append(result)
        rows.extend(clean_rows(result["values"].get("rows")))
    reading = {
        "cohort_month": cohort["values"].get("cohort_month"),
        "cohort_year": cohort["values"].get("cohort_year"),
        "rows": rows,
    }
    return {
        "reading": reading,
        "mode": mode,
        "json_valid": all(r["json_valid"] for r in results),
        "errors": [e for r in results for e in r["errors"]],
        "raw": [r["raw"] for r in results],
        "notes": notes,
        "aligned": aligned,
    }


def read_cover(photo: Path, model: str, *, think: bool = False) -> dict[str, Any]:
    result = _call(photo, cover_schema(), cover_prompt(), model, think, FULL_MAX_SIDE, "register-cover")  # fmt: skip
    return {"cover": result["values"], "json_valid": result["json_valid"],
            "errors": result["errors"], "raw": [result["raw"]]}  # fmt: skip
