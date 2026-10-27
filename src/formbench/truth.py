from __future__ import annotations

import datetime as dt
import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from formbench.schema_gen import JSON_TYPES

MAX_COUNT = 150

# Made-up names only: no real facility should ever appear in constructed data.
PLACES = ["Amani", "Baraka", "Jamhuri", "Kilele", "Mwangaza", "Neema", "Tumaini", "Upendo"]
AREAS = ["North", "South", "East", "West", "Central", "Hill", "Market", "River"]
KINDS = ["Health Centre", "Dispensary", "Clinic", "Health Post", "Hospital"]

YEARS = (2024, 2025)


def make_period(period_type: str, rng: random.Random) -> str:
    """A valid DHIS2 period identifier for the data set's periodType."""
    year = rng.randint(*YEARS)
    if period_type == "Daily":
        day = dt.date(year, 1, 1) + dt.timedelta(days=rng.randrange(365))
        return day.strftime("%Y%m%d")
    if period_type == "Weekly":
        return f"{year}W{rng.randint(1, 52)}"
    if period_type == "Monthly":
        return f"{year}{rng.randint(1, 12):02d}"
    if period_type == "Quarterly":
        return f"{year}Q{rng.randint(1, 4)}"
    if period_type == "Yearly":
        return str(year)
    raise ValueError(f"Unsupported periodType {period_type} for generated forms")


def make_truth(
    schema: dict[str, Any], mapping: dict[str, Any], seed: int, blank_ratio: float
) -> dict[str, Any]:
    """Ground truth in the extraction schema's shape: header values, counts and blanks."""
    if not 0 <= blank_ratio <= 1:
        raise ValueError("blank_ratio must be between 0 and 1")
    rng = random.Random(f"truth-{seed}")
    fields = mapping["fields"]
    truth: dict[str, Any] = {}
    for key, prop in schema["properties"].items():
        if key == "org_unit_name":
            truth[key] = f"{rng.choice(PLACES)} {rng.choice(AREAS)} {rng.choice(KINDS)}"
        elif key == "period":
            truth[key] = make_period(mapping["dataSet"]["periodType"], rng)
        elif key == "attribute_option":
            truth[key] = rng.choice([o for o in prop["enum"] if o is not None])
        elif JSON_TYPES[fields[key]["valueType"]] not in ("integer", "number"):
            raise ValueError(f"Cannot generate a value for non-numeric field {key}")
        elif rng.random() < blank_ratio:
            truth[key] = None
        else:
            truth[key] = rng.randint(0, MAX_COUNT)
    return truth


def draw_values_sheet(
    truth: dict[str, Any], mapping: dict[str, Any], title: str
) -> list[Image.Image]:
    """Pages listing what to write in each box, in the order of the form."""
    lines = [
        f"Facility name: {truth['org_unit_name']}",
        f"Period: {truth['period']}",
    ]
    if "attribute_option" in truth:
        lines.append(f"Attribute option: {truth['attribute_option']}")
    lines.append("")
    for key, field in mapping["fields"].items():
        value = truth[key]
        lines.append(f"{field['label']}: {'leave empty' if value is None else value}")

    width, height, margin, line_h = 2480, 3508, 200, 64
    font, heading = ImageFont.load_default(size=40), ImageFont.load_default(size=60)
    per_page = (height - 2 * margin - 160) // line_h
    pages = []
    for start in range(0, len(lines), per_page):
        page = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin), f"{title}: values to copy", font=heading, fill=0)
        for i, line in enumerate(lines[start : start + per_page]):
            draw.text((margin, margin + 160 + i * line_h), line, font=font, fill=0)
        pages.append(page)
    return pages
