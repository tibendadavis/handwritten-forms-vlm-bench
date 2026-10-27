from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DPI = 300
REGISTER_PDF = Path("data/private/samples/R-06_EPI Register.pdf")
TEMPLATE_DIR = Path("data/private/register")

ROWS = 17
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["AA", "AB", "AC"]


class TemplateError(ValueError):
    """The rendered page does not have the R-06 grid we expect."""


def _lines(mask: np.ndarray, axis: int, min_len: int) -> list[tuple[int, int]]:
    """(position, length) of each straight line in a binary mask, merged across thickness."""
    profile = mask.sum(axis=axis) // 255
    out: list[list[int]] = []
    for i in np.where(profile >= min_len)[0]:
        if out and i - out[-1][-1] <= 3:
            out[-1].append(int(i))
        else:
            out.append([int(i)])
    return [((g[0] + g[-1]) // 2, int(profile[g].max())) for g in out]


def detect_grid(gray: np.ndarray) -> dict[str, Any]:
    """Columns A..AC, the 17 child rows (with the C/F sub-line) and the header band."""
    ink = (gray < 160).astype(np.uint8) * 255
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))

    verticals = [x for x, _ in _lines(v, 0, gray.shape[0] // 4)]
    if len(verticals) != len(LETTERS) + 1:
        raise TemplateError(f"Expected {len(LETTERS) + 1} column lines, found {len(verticals)}")
    table_w = verticals[-1] - verticals[0]
    full = [y for y, length in _lines(h, 1, 200) if length > 0.9 * table_w]
    short = [y for y, length in _lines(h, 1, 200) if length <= 0.9 * table_w]
    # Letter row = the two full lines closest together; child rows start below it.
    gaps = [b - a for a, b in zip(full, full[1:])]
    letters_top = full[gaps.index(min(gaps))]
    body = [y for y in full if y > letters_top]
    if len(body) != ROWS + 1:
        raise TemplateError(f"Expected {ROWS} child rows, found {len(body) - 1}")

    rows = []
    for top, bottom in zip(body, body[1:]):
        inside = [y for y in short if top + 10 < y < bottom - 10]
        rows.append(
            {"y0": top, "y1": bottom, "split": inside[0] if inside else (top + bottom) // 2}
        )
    columns = {letter: [x0, x1] for letter, x0, x1 in zip(LETTERS, verticals, verticals[1:])}
    return {
        "columns": columns,
        "rows": rows,
        "header": {"y0": full[0], "y1": body[0]},
    }


def render_page(pdf: Path = REGISTER_PDF) -> tuple[np.ndarray, Any]:
    import pypdfium2 as pdfium

    page = pdfium.PdfDocument(str(pdf))[0]
    image = np.array(page.render(scale=DPI / 72).to_pil().convert("L"))
    return image, page


def _label_box(page: Any, word: str, height_pt: float) -> list[int]:
    """Pixel box of a printed label, from the PDF text layer (PDF y grows upwards)."""
    textpage = page.get_textpage()
    found = textpage.search(word, match_case=True, match_whole_word=True).get_next()
    if not found:
        raise TemplateError(f"Label {word!r} not found on the register page")
    start, count = found
    boxes = [textpage.get_charbox(i) for i in range(start, start + count)]
    scale = DPI / 72
    left, right = min(b[0] for b in boxes), max(b[2] for b in boxes)
    bottom, top = min(b[1] for b in boxes), max(b[3] for b in boxes)
    return [round(left * scale), round((height_pt - top) * scale), round(right * scale),
            round((height_pt - bottom) * scale)]  # fmt: skip


def build_template(pdf: Path = REGISTER_PDF) -> tuple[np.ndarray, dict[str, Any]]:
    image, page = render_page(pdf)
    layout = detect_grid(image)
    height_pt = page.get_size()[1]
    month, year = _label_box(page, "Month", height_pt), _label_box(page, "Year", height_pt)
    top = layout["header"]["y0"] - 110
    # Handwriting goes right of each printed label, above the table.
    layout["month_box"] = [month[2] + 15, top, year[0] - 40, layout["header"]["y0"] - 5]
    layout["year_box"] = [year[2] + 15, top, year[2] + 520, layout["header"]["y0"] - 5]
    layout["page"] = {"width": image.shape[1], "height": image.shape[0], "dpi": DPI}
    return image, layout


def load_template(folder: Path = TEMPLATE_DIR) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(str(folder / "template.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"No register template in {folder}; run `uv run register-template`.")
    return image, json.loads((folder / "template.json").read_text())


def overlay(image: np.ndarray, layout: dict[str, Any]) -> np.ndarray:
    """The detected layout drawn over the template, to check detection by eye."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for letter, (x0, x1) in layout["columns"].items():
        cv2.putText(out, letter, (x0 + 5, layout["header"]["y1"] - 10), 0, 1.2, (0, 0, 255), 3)
        for row in layout["rows"]:
            cv2.rectangle(out, (x0 + 4, row["y0"] + 4), (x1 - 4, row["y1"] - 4), (0, 160, 0), 2)
    for key in ("month_box", "year_box"):
        x0, y0, x1, y1 = layout[key]
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R-06 template from the blank PDF.")
    parser.add_argument("--pdf", type=Path, default=REGISTER_PDF)
    parser.add_argument("--out", type=Path, default=TEMPLATE_DIR)
    args = parser.parse_args()

    try:
        image, layout = build_template(args.pdf)
    except TemplateError as error:
        raise SystemExit(f"register-template: {error}") from None
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "template.png"), image)
    cv2.imwrite(str(args.out / "template-overlay.jpg"), overlay(image, layout))
    (args.out / "template.json").write_text(json.dumps(layout, indent=2) + "\n")
    print(
        f"R-06 template: {len(layout['rows'])} rows x {len(layout['columns'])} columns -> {args.out}/"
    )


if __name__ == "__main__":
    main()
