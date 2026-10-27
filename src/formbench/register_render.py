"""Draw synthetic register entries as handwriting on the real R-06 page, then 'photograph' it.

Handwriting: OFL handwriting fonts, one writer style per page (font, size, slant, ink),
per-glyph jitter, tiny stacked dates in the narrow dose columns, occasional crossed-out
dates. Photos: perspective, lighting, shadow, curl, blur, noise and JPEG in three tiers.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("data/fonts")
FONT_BASE = "https://github.com/google/fonts/raw/main/ofl/"
# (path in google/fonts, neatness 0..1). All SIL Open Font License.
FONTS = [
    ("patrickhand/PatrickHand-Regular.ttf", 0.9),
    ("kalam/Kalam-Regular.ttf", 0.85),
    ("handlee/Handlee-Regular.ttf", 0.85),
    ("caveat/Caveat%5Bwght%5D.ttf", 0.8),
    ("architectsdaughter/ArchitectsDaughter-Regular.ttf", 0.8),
    ("indieflower/IndieFlower-Regular.ttf", 0.75),
    ("gochihand/GochiHand-Regular.ttf", 0.7),
    ("shadowsintolight/ShadowsIntoLight.ttf", 0.6),
    ("coveredbyyourgrace/CoveredByYourGrace.ttf", 0.6),
    ("reeniebeanie/ReenieBeanie.ttf", 0.45),
    ("nothingyoucoulddo/NothingYouCouldDo.ttf", 0.4),
    ("zeyada/Zeyada.ttf", 0.35),
    ("dawningofanewday/DawningofaNewDay.ttf", 0.3),
]
INKS = {"blue": (28, 45, 140), "black": (28, 28, 34), "dark blue": (20, 30, 90)}
QUALITY_WEIGHTS = {"scan": 0.2, "good": 0.5, "poor": 0.3}


def ensure_fonts(folder: Path = FONT_DIR) -> list[tuple[Path, float]]:
    """Download the handwriting fonts once into the gitignored cache."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for path, neatness in FONTS:
        target = folder / Path(path).name.replace("%5B", "[").replace("%5D", "]")
        if not target.exists():
            with urllib.request.urlopen(FONT_BASE + path, timeout=60) as response:  # noqa: S310
                target.write_bytes(response.read())
        out.append((target, neatness))
    return out


@dataclass
class Writer:
    font: Path
    neatness: float
    size: int
    slant: float
    ink: tuple[int, int, int]
    stack: int  # lines used for a date in a narrow column: 2 or 3


def make_writer(rng: random.Random, fonts: list[tuple[Path, float]]) -> Writer:
    font, neatness = rng.choice(fonts)
    return Writer(
        font=font,
        neatness=neatness,
        size=rng.randint(52, 64),
        slant=rng.uniform(-0.18, 0.12),
        ink=INKS[rng.choice(["blue", "blue", "black", "dark blue"])],
        stack=rng.choice([2, 2, 3]),
    )


def _glyphs(text: str, writer: Writer, size: int, rng: random.Random) -> Image.Image:
    """One line of text, glyph by glyph with jitter, as an alpha mask."""
    jitter = 1.2 - writer.neatness
    font = ImageFont.truetype(str(writer.font), size)
    pieces, width = [], 0
    for ch in text:
        if ch == " ":
            width += int(size * rng.uniform(0.25, 0.4))
            continue
        glyph_size = max(8, int(size * rng.uniform(1 - 0.08 * jitter, 1 + 0.08 * jitter)))
        f = font if glyph_size == size else ImageFont.truetype(str(writer.font), glyph_size)
        box = f.getbbox(ch)
        w, h = box[2] - box[0] + 8, size * 2
        tile = Image.new("L", (w, h), 0)
        ImageDraw.Draw(tile).text((4 - box[0], size // 2), ch, font=f, fill=255)
        tile = tile.rotate(rng.uniform(-6, 6) * jitter, resample=Image.BICUBIC, expand=False)
        dy = int(rng.gauss(0, 2.5 * jitter))
        pieces.append((tile, width, dy))
        width += int((box[2] - box[0]) * rng.uniform(0.95, 1.12)) + int(rng.uniform(0, 3))
    canvas = Image.new("L", (max(width + 8, 1), size * 2 + 10), 0)
    for tile, x, dy in pieces:
        canvas.paste(tile, (x, 5 + dy), tile)
    bbox = canvas.getbbox()
    if not bbox:
        return canvas
    canvas = canvas.crop(bbox)
    shear = writer.slant + rng.uniform(-0.04, 0.04)
    w, h = canvas.size
    return canvas.transform((w + int(abs(shear) * h), h), Image.AFFINE,
                            (1, shear, -shear * h if shear > 0 else 0, 0, 1, 0), Image.BICUBIC)  # fmt: skip


def write(page: Image.Image, text: str, box: tuple[int, int, int, int], writer: Writer,
          rng: random.Random, lines: list[str] | None = None, scale: float = 1.0) -> None:  # fmt: skip
    """Handwrite text into a box (x0, y0, x1, y1), shrinking to fit like a person would."""
    x0, y0, x1, y1 = box
    lines = lines or [text]
    size = int(writer.size * scale)
    masks = [_glyphs(line, writer, size, rng) for line in lines if line]
    if not masks:
        return
    pad = 6
    w = max(m.size[0] for m in masks)
    h = sum(m.size[1] for m in masks) + 4 * (len(masks) - 1)
    fit = min(1.0, (x1 - x0 - 2 * pad) / max(w, 1), (y1 - y0 - 2 * pad) / max(h, 1))
    y = y0 + max(pad, int((y1 - y0 - h * fit) / 2 + rng.uniform(-4, 4)))
    pressure = rng.uniform(0.78, 0.97)
    for mask in masks:
        mw, mh = max(1, int(mask.size[0] * fit)), max(1, int(mask.size[1] * fit))
        mask = mask.resize((mw, mh), Image.LANCZOS).point(lambda v: int(v * pressure))
        x = x0 + pad + int(rng.uniform(0, max(0, x1 - x0 - 2 * pad - mw) * 0.6))
        page.paste(Image.new("RGB", (mw, mh), writer.ink), (x, y), mask)
        y += mh + 4


def _strike(page: Image.Image, box: tuple[int, int, int, int], writer: Writer, rng: random.Random) -> None:  # fmt: skip
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(page)
    for _ in range(rng.choice([1, 2])):
        ya, yb = y0 + (y1 - y0) * rng.uniform(0.3, 0.7), y0 + (y1 - y0) * rng.uniform(0.3, 0.7)
        draw.line([(x0 + 4, ya), (x1 - 4, yb)], fill=writer.ink, width=4)


def _cell(layout, letter: str, row: dict[str, int], part: str = "all") -> tuple[int, int, int, int]:
    x0, x1 = layout["columns"][letter]
    y0, y1 = row["y0"], row["y1"]
    if part == "top":
        y1 = row["split"]
    elif part == "bottom":
        y0 = row["split"]
    return x0 + 3, y0 + 3, x1 - 3, y1 - 3


def _date_lines(text: str, narrow: bool, writer: Writer) -> list[str]:
    if not narrow:
        return [text]
    sep = "/" if "/" in text else "-"
    d, m, y = text.split(sep)
    return [f"{d}{sep}{m}", y] if writer.stack == 2 else [d, m, y]


def render_page(template: np.ndarray, layout: dict[str, Any], reading: dict[str, Any],
                private: dict[str, dict[str, Any]], writer: Writer, rng: random.Random,
                date_style_fmt) -> Image.Image:  # fmt: skip
    """The register page with everything written so far, on slightly off-white paper."""
    from formbench.register_spec import DOSE_COLUMNS

    paper = np.dstack([template] * 3).astype(np.float32)
    tint = np.array([250, 247, 238], np.float32) / 255
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 3, template.shape)[..., None]
    page = Image.fromarray(np.clip(paper * tint + noise, 0, 255).astype(np.uint8))

    mx0, my0, mx1, my1 = layout["month_box"]
    write(page, reading["cohort_month"], (mx0, my0, mx1, my1), writer, rng, scale=1.3)
    write(page, reading["cohort_year"], tuple(layout["year_box"]), writer, rng, scale=1.3)
    letters = {c.key: c.letter for c in DOSE_COLUMNS}
    for index, row in enumerate(reading["rows"]):
        if index >= len(layout["rows"]):
            break
        cell = layout["rows"][index]
        extra = private.get(row["serial"], {})
        write(page, row["serial"], _cell(layout, "A", cell), writer, rng)
        if extra.get("first_visit"):
            write(page, date_style_fmt(extra["first_visit"]), _cell(layout, "B", cell), writer, rng, scale=0.8)  # fmt: skip
        if row.get("birth_date"):
            write(page, row["birth_date"], _cell(layout, "C", cell, "top"), writer, rng, scale=0.7)
        if row.get("age"):
            write(page, row["age"], _cell(layout, "C", cell, "bottom"), writer, rng, scale=0.7)
        write(page, row.get("card_number") or "", _cell(layout, "D", cell), writer, rng, scale=0.85)
        write(page, extra.get("names", ""), _cell(layout, "E", cell), writer, rng, scale=0.85)
        write(
            page, extra.get("guardian", ""), _cell(layout, "F", cell, "top"), writer, rng, scale=0.7
        )
        write(
            page, extra.get("phone", ""), _cell(layout, "F", cell, "bottom"), writer, rng, scale=0.7
        )
        write(page, row["sex"], _cell(layout, "G", cell), writer, rng)
        write(page, extra.get("village", ""), _cell(layout, "H", cell), writer, rng, scale=0.8)
        for key, letter in letters.items():
            text = row.get(key)
            if not text:
                continue
            box = _cell(layout, letter, cell)
            narrow = box[2] - box[0] < 110
            if extra.get("crossed") == key:
                wrong = _wrong_date(text, rng)
                top = (box[0], box[1], box[2], (box[1] + box[3]) // 2)
                bottom = (box[0], (box[1] + box[3]) // 2, box[2], box[3])
                write(page, wrong, top, writer, rng, _date_lines(wrong, narrow, writer), 0.6)
                _strike(page, top, writer, rng)
                write(page, text, bottom, writer, rng, _date_lines(text, narrow, writer), 0.6)
                continue
            write(page, text, box, writer, rng, _date_lines(text, narrow, writer), 0.62 if narrow else 0.8)  # fmt: skip
    return page.filter(ImageFilter.GaussianBlur(0.5))


def _wrong_date(text: str, rng: random.Random) -> str:
    sep = "/" if "/" in text else "-"
    d, m, y = text.split(sep)
    day = int(d) + rng.choice([-2, -1, 1, 2])
    return sep.join([f"{min(max(day, 1), 28):0{len(d)}d}", m, y])


def render_cover(cover: dict[str, Any], writer: Writer, rng: random.Random) -> Image.Image:
    """A register cover like the R-06 one: printed labels, handwritten values."""
    w, h = 3508, 2480
    page = Image.new("RGB", (w, h), (250, 247, 238))
    draw = ImageDraw.Draw(page)
    bold = ImageFont.load_default(size=110)
    draw.text((w // 2, 620), "MINISTRY OF HEALTH", font=bold, fill=(20, 20, 20), anchor="mm")
    draw.text((w // 2, 860), "IMMUNISATION (EPI) REGISTER", font=ImageFont.load_default(size=90), fill=(20, 20, 20), anchor="mm")  # fmt: skip
    draw.text((w // 2, 1040), "R-06", font=ImageFont.load_default(size=90), fill=(20, 20, 20), anchor="mm")  # fmt: skip
    label = ImageFont.load_default(size=48)
    fields = [("Facility Name:", "facility_name"), ("MFL Code:", "mfl_code"),
              ("Facility Type:", "facility_type"), ("District:", "district"), ("Region:", "region")]  # fmt: skip
    x0, x1, lx = 720, 2860, 1160
    y = 1300
    for text, key in fields:
        draw.rectangle([x0, y, x1, y + 110], outline=(20, 20, 20), width=4)
        draw.line([(lx, y), (lx, y + 110)], fill=(20, 20, 20), width=4)
        draw.text((x0 + 20, y + 55), text, font=label, fill=(20, 20, 20), anchor="lm")
        write(
            page, cover.get(key) or "", (lx + 10, y + 5, x1 - 10, y + 105), writer, rng, scale=1.1
        )
        y += 110
    return page


def photograph(page: Image.Image, quality: str, rng: random.Random) -> Image.Image:
    """Simulate a phone photo of a page lying on a table."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    img = np.array(page).astype(np.float32)
    h, w = img.shape[:2]
    strength = {"scan": 0.0, "good": 0.03, "poor": 0.07}[quality]
    if quality == "poor":  # gentle page curl
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        ys += 18 * np.sin(xs / w * math.pi) * rng.uniform(0.5, 1.0)
        img = cv2.remap(img, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    frame_w = int(w * (1.0 if quality == "scan" else rng.uniform(1.08, 1.18)))
    frame_h = int(frame_w * h / w * rng.uniform(1.0, 1.05))
    ox, oy = (frame_w - w) / 2, (frame_h - h) / 2
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    moved = corners + [ox, oy] + nrng.uniform(-strength, strength, (4, 2)) * [w, h]
    if quality == "scan":
        angle = math.radians(rng.uniform(-0.6, 0.6))
        c, s = math.cos(angle), math.sin(angle)
        moved = np.float32([[c * x - s * y, s * x + c * y] for x, y in corners - [w / 2, h / 2]]) + [frame_w / 2, frame_h / 2]  # fmt: skip
    matrix = cv2.getPerspectiveTransform(corners, np.float32(moved))
    table = np.array([rng.randint(90, 140), rng.randint(70, 110), rng.randint(50, 80)], np.float32)
    background = (255, 255, 255) if quality == "scan" else tuple(float(v) for v in table)
    out = cv2.warpPerspective(img, matrix, (frame_w, frame_h), borderValue=background)

    if quality != "scan":
        ys, xs = np.mgrid[0:frame_h, 0:frame_w].astype(np.float32)
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        light = 1 - 0.12 * (gx * (xs / frame_w - 0.5) + gy * (ys / frame_h - 0.5)) * 2
        light -= 0.10 * (((xs / frame_w - 0.5) ** 2 + (ys / frame_h - 0.5) ** 2) * 2)
        if quality == "poor":  # a soft shadow across part of the page
            edge = rng.uniform(0.2, 0.8) * frame_w + (ys - frame_h / 2) * rng.uniform(-0.4, 0.4)
            light *= np.where(xs < edge, rng.uniform(0.7, 0.82), 1.0)
            light = cv2.GaussianBlur(light, (0, 0), 60)
        out *= light[..., None]
        sigma = {"good": 0.8, "poor": 1.4}[quality]
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    out += nrng.normal(0, {"scan": 2, "good": 4, "poor": 7}[quality], out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    photo = Image.fromarray(out)
    if quality == "poor":
        photo = photo.resize((int(frame_w * 0.75), int(frame_h * 0.75)), Image.LANCZOS)
    return photo


def render_scenario(folder: Path, sim: dict[str, Any], truth: list[dict[str, Any]],
                    cover: dict[str, Any], styles: list[str], *, quality: str, seed: int) -> None:  # fmt: skip
    """Write cover.jpg and each week's page photos; record file and quality in the truth."""
    from formbench.register_synth import fmt_date, private_fields
    from formbench.register_template import load_template

    rng = random.Random(f"render-{seed}")
    fonts = ensure_fonts()
    template, layout = load_template()
    writers = [make_writer(rng, fonts) for _ in sim["book"]]

    def pick_quality() -> str:
        if quality != "mixed":
            return quality
        return rng.choices(list(QUALITY_WEIGHTS), weights=list(QUALITY_WEIGHTS.values()))[0]

    cover_quality = pick_quality()
    photograph(render_cover(cover, make_writer(rng, fonts), rng), cover_quality, rng).save(folder / "cover.jpg", quality=88)  # fmt: skip
    for index, week in enumerate(truth, start=1):
        week_dir = folder / f"week-{index:02d}"
        week_dir.mkdir()
        week["cover"] = "cover.jpg"
        for photo in week["photos"]:
            i = photo["page_index"]
            page = sim["book"][i]
            style = styles[i]

            def as_written(day: dt.date, style=style) -> str:
                return fmt_date(day, style)

            image = render_page(template, layout, photo["reading"], private_fields(page),
                                writers[i], rng, as_written)  # fmt: skip
            photo["quality"] = pick_quality()
            photo["file"] = f"week-{index:02d}/page-{photo['page_id']}.jpg"
            photograph(image, photo["quality"], rng).save(folder / photo["file"], quality=88 if photo["quality"] != "poor" else 72)  # fmt: skip
