from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from formbench.schema_gen import (
    JSON_TYPES,
    Field,
    build_extraction_schema,
    build_fields,
    build_mapping,
)
from formbench.truth import draw_values_sheet, make_truth

# A4 at 300 DPI. All layout coordinates are page pixels at this size.
DPI = 300
PAGE_W, PAGE_H = 2480, 3508

# Solid squares at the page corners; align.py finds them in photos.
MARKER = 100
MARKER_INSET = 100

# Printable area, kept clear of the markers.
CONTENT_X0, CONTENT_X1 = 260, PAGE_W - 260
CONTENT_Y0, CONTENT_Y1 = 260, PAGE_H - 260

LINE = 3
PAD = 20
ROW_H = 140
HEADER_H = 130
HEADER_LABEL_W = 420
LABEL_COL_MIN = 700
MIN_CELL_W, MAX_CELL_W = 170, 320

# Header boxes in printed order; keys match the extraction schema.
HEADER_LABELS = {
    "org_unit_name": "Facility name",
    "period": "Period",
    "attribute_option": "Attribute option",
}

# How a person writes each DHIS2 period type, printed as a hint next to the period box.
PERIOD_HINTS = {
    "Daily": "YYYYMMDD",
    "Weekly": "YYYYWn",
    "Monthly": "YYYYMM",
    "Quarterly": "YYYYQn",
    "Yearly": "YYYY",
}

NUMERIC = ("integer", "number")


class FormError(ValueError):
    """A form cannot be generated as requested; the message says what to change."""


def _font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled font: identical rendering on every OS, no system font lookup.
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line] if line else lines


def _box(x0: int, y0: int, x1: int, y1: int) -> dict[str, int]:
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def select_elements(ds: dict[str, Any], n: int, section: str | None, seed: int) -> list[Field]:
    """Fields of n numeric data elements sampled with the seed, kept in form order."""
    fields = build_fields(ds)
    if section is not None:
        sections = sorted({f.section for f in fields if f.section})
        if section not in sections:
            raise FormError(f"No section '{section}'. Sections: {', '.join(sections)}")
        fields = [f for f in fields if f.section == section]
    numeric = [f for f in fields if JSON_TYPES[f.value_type] in NUMERIC]
    skipped = sorted({f.label for f in fields if f not in numeric})
    if skipped:
        print(f"Skipping non-numeric fields (cells hold digits only): {', '.join(skipped)}")
    elements = list(dict.fromkeys(f.data_element for f in numeric))
    if not elements:
        raise FormError("No numeric data elements to print.")
    chosen = set(random.Random(seed).sample(elements, min(n, len(elements))))
    return [f for f in numeric if f.data_element in chosen]


def _tables(ds: dict[str, Any], fields: list[Field]) -> list[dict[str, Any]]:
    """One table per category combo: rows are data elements, columns are option combos."""
    names: dict[str, str] = {}
    combos: dict[str, dict[str, Any]] = {}
    for dse in ds["dataSetElements"]:
        de = dse["dataElement"]
        names[de["id"]] = de.get("formName") or de["name"]
        combos[de["id"]] = dse.get("categoryCombo") or de["categoryCombo"]
    keys = {(f.data_element, f.category_option_combo): f.key for f in fields}
    tables: dict[str, dict[str, Any]] = {}
    for de_id in dict.fromkeys(f.data_element for f in fields):
        combo = combos[de_id]
        cocs = sorted(combo["categoryOptionCombos"], key=lambda c: c["name"])
        table = tables.setdefault(
            combo["id"],
            {
                "columns": ["Value" if c["name"] == "default" else c["name"] for c in cocs],
                "rows": [],
            },
        )
        table["rows"].append((names[de_id], [keys[(de_id, c["id"])] for c in cocs]))
    return list(tables.values())


def draw_form(
    ds: dict[str, Any], fields: list[Field], header_keys: list[str]
) -> tuple[Image.Image, dict[str, Any]]:
    """Render the one-page form and return it with its layout (every box in page pixels)."""
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(img)
    title, text, small = _font(72), _font(40), _font(32)

    far_x, far_y = PAGE_W - MARKER_INSET - MARKER, PAGE_H - MARKER_INSET - MARKER
    corners = [
        (MARKER_INSET, MARKER_INSET),
        (far_x, MARKER_INSET),
        (far_x, far_y),
        (MARKER_INSET, far_y),
    ]
    markers = [_box(x, y, x + MARKER, y + MARKER) for x, y in corners]
    for x, y in corners:
        draw.rectangle([x, y, x + MARKER - 1, y + MARKER - 1], fill=0)

    y = CONTENT_Y0
    draw.text((CONTENT_X0, y), ds["name"], font=title, fill=0)
    y += 110
    hint = "Write one number per box. Leave a box empty when there is nothing to report."
    draw.text((CONTENT_X0, y), hint, font=small, fill=0)
    y += 90

    header: dict[str, dict[str, int]] = {}
    for key in header_keys:
        label = HEADER_LABELS[key]
        x1 = CONTENT_X1
        if key == "period":
            label = f"{label} ({PERIOD_HINTS.get(ds['periodType'], ds['periodType'])})"
            x1 = CONTENT_X0 + HEADER_LABEL_W + 700
        draw.text((CONTENT_X0, y + HEADER_H // 2), label, font=text, fill=0, anchor="lm")
        x0 = CONTENT_X0 + HEADER_LABEL_W
        draw.rectangle([x0, y, x1, y + HEADER_H], outline=0, width=LINE)
        header[key] = _box(x0, y, x1, y + HEADER_H)
        y += HEADER_H + 30
    y += 40

    width = CONTENT_X1 - CONTENT_X0
    cells: dict[str, dict[str, int]] = {}
    placed = 0
    for table in _tables(ds, fields):
        columns = table["columns"]
        label_w = max(LABEL_COL_MIN, width - len(columns) * MAX_CELL_W)
        cell_w = (width - label_w) // len(columns)
        if cell_w < MIN_CELL_W:
            raise FormError(f"{len(columns)} category option combos do not fit across one A4 page.")
        xs = [CONTENT_X0 + label_w + i * cell_w for i in range(len(columns) + 1)]

        heads = [_wrap(draw, c, small, cell_w - 2 * PAD) for c in columns]
        head_h = max(ROW_H, max(len(h) for h in heads) * 40 + 2 * PAD)
        rows = [
            (_wrap(draw, label, text, label_w - 2 * PAD), keys) for label, keys in table["rows"]
        ]
        if y + head_h + ROW_H > CONTENT_Y1:
            raise FormError(_too_long(placed))
        for x0, x1, lines in zip(xs, xs[1:], heads):
            draw.rectangle([x0, y, x1, y + head_h], outline=0, width=LINE)
            draw.multiline_text(
                ((x0 + x1) // 2, y + head_h // 2),
                "\n".join(lines),
                font=small,
                fill=0,
                anchor="mm",
                align="center",
            )
        y += head_h

        for lines, keys in rows:
            row_h = max(ROW_H, len(lines) * 48 + 2 * PAD)
            if y + row_h > CONTENT_Y1:
                raise FormError(_too_long(placed))
            draw.rectangle([CONTENT_X0, y, xs[0], y + row_h], outline=0, width=LINE)
            draw.multiline_text(
                (CONTENT_X0 + PAD, y + row_h // 2), "\n".join(lines), font=text, fill=0, anchor="lm"
            )
            for key, x0, x1 in zip(keys, xs, xs[1:]):
                draw.rectangle([x0, y, x1, y + row_h], outline=0, width=LINE)
                cells[key] = _box(x0, y, x1, y + row_h)
            y += row_h
            placed += 1
        y += 60

    layout = {
        "page": {"width": PAGE_W, "height": PAGE_H, "dpi": DPI},
        "markers": markers,
        "header": header,
        "fields": cells,
    }
    return img, layout


def _too_long(placed: int) -> str:
    if placed == 0:
        return "Not even one data element fits on the page."
    return f"The selection does not fit on one A4 page; try --elements {placed}."


def make_form(
    schema_dir: Path,
    *,
    elements: int = 6,
    section: str | None = None,
    seed: int = 1,
    blank_ratio: float = 0.25,
    out: Path = Path("forms"),
) -> Path:
    metadata = schema_dir / "metadata.json"
    if not metadata.exists():
        raise FormError(f"{metadata} not found; regenerate the schema with build-schema.")
    ds = json.loads(metadata.read_text())

    fields = select_elements(ds, elements, section, seed)
    schema = build_extraction_schema(ds, fields)
    mapping = build_mapping(ds, fields)
    header_keys = [k for k in HEADER_LABELS if k in schema["properties"]]
    img, layout = draw_form(ds, fields, header_keys)
    truth = make_truth(schema, mapping, seed, blank_ratio)
    sheet = draw_values_sheet(truth, mapping, ds["name"])

    folder = out / f"{schema_dir.name}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "form.png", dpi=(DPI, DPI))
    img.save(folder / "form.pdf", "PDF", resolution=DPI)
    (folder / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    (folder / "extract.schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    )
    (folder / "mapping.yaml").write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True)
    )
    (folder / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    sheet[0].save(
        folder / "values_sheet.pdf", "PDF", resolution=DPI, save_all=True, append_images=sheet[1:]
    )
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable one-page form.")
    parser.add_argument("schema_dir", type=Path, help="schema folder, e.g. schemas/child_health")
    parser.add_argument("--elements", type=int, default=6, help="data elements to print")
    parser.add_argument("--section", help="only pick data elements from this section")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blank-ratio", type=float, default=0.25, help="share of empty cells")
    parser.add_argument("--out", type=Path, default=Path("forms"))
    args = parser.parse_args()

    try:
        folder = make_form(
            args.schema_dir,
            elements=args.elements,
            section=args.section,
            seed=args.seed,
            blank_ratio=args.blank_ratio,
            out=args.out,
        )
    except ValueError as error:
        raise SystemExit(f"make-form: {error}") from None
    print(f"Form written to {folder}/")


if __name__ == "__main__":
    main()
