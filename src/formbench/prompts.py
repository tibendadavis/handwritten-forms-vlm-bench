"""Prompts versioned in Langfuse prompt management.

The text in code is only version 1: the first run that finds no prompt saves it with the
production label. After that, prompts are edited in Langfuse and each run picks a version.
A version that was asked for and does not exist stops the run; it never falls back to another.

The code defaults below are the best prompts found so far, kept word for word.
"""

from __future__ import annotations

from typing import Any

from langfuse.api import NotFoundError

# One register page photo. No variables.
REGISTER_PAGE_PROMPT = (
    "This is a photo of one page of a handwritten immunisation register. Columns are labelled "
    "with letters A to AC in the row just under the header. Each child is one row across the "
    "page. The dose columns I to Y are narrow, so the dates in them are written sideways: "
    "rotated 90 degrees, reading from bottom to top, in the form DD-MM-YY (for example "
    "26-08-26). Each sideways date starts at the bottom of its own row's cell and may run over "
    "the line into the row above; it belongs to the row where it starts. Read each sideways "
    "date as if the page were turned. Older children have dates in many dose columns side by "
    "side.\n"
    "Return cohort_month and cohort_year from the page header, and rows with these keys:\n"
    "- serial: column A Serial Number\n"
    "- card_number: column D Child Card Number\n"
    "- sex: column G Sex, M or F as written\n"
    "- birth_date: column C Birth Date, top line of the cell, exactly as written\n"
    "- age: column C Age, bottom line of the cell, exactly as written (e.g. 7, 3/12, 6/52, 5/7)\n"
    "- bcg: column I BCG date, exactly as written\n"
    "- polio: column J Polio date, exactly as written\n"
    "- opv0: column K OPV 0 (birth) date, exactly as written\n"
    "- opv1: column L OPV 1 date, exactly as written\n"
    "- opv2: column M OPV 2 date, exactly as written\n"
    "- opv3: column N OPV 3 date, exactly as written\n"
    "- penta1: column O Pentavalent 1 date, exactly as written\n"
    "- penta2: column P Pentavalent 2 date, exactly as written\n"
    "- penta3: column Q Pentavalent 3 date, exactly as written\n"
    "- ipv1: column R IPV 1 date, exactly as written\n"
    "- ipv2: column S IPV 2 date, exactly as written\n"
    "- pcv1: column T PCV 1 date, exactly as written\n"
    "- pcv2: column U PCV 2 date, exactly as written\n"
    "- pcv3: column V PCV 3 date, exactly as written\n"
    "- rota1: column W Rotavirus 1 date, exactly as written\n"
    "- mcv1: column X MCV 1 date, exactly as written\n"
    "- mcv2: column Y MCV 2 date, exactly as written\n"
    "\n"
    "Rules:\n"
    "- Every row object must contain all 22 keys. For each child row, check every dose column "
    "from I to Y one by one before moving to the next row. Do not stop after column K.\n"
    "- If a cell has writing in it, give your best reading of it, even if unclear. Use null "
    "only when the cell has no writing at all.\n"
    "- Each date belongs only under the column it is written in. Many dose cells are empty; "
    "never fill a cell from a neighbouring cell or row.\n"
    "- The birth date is the top line of column C, not column B (date of first visit).\n"
    "- Copy every value as handwritten; do not reformat dates.\n"
    "- One object per child row that has anything written in it; skip empty rows.\n"
    "- Do not transcribe child names, mother/guardian names, phone numbers or villages.\n"
    "- Output only the JSON object."
)

# A cv-form section close-up. Variables: section (letter), title, fields.
SECTION_PROMPT = (
    'This is a photo of one section of a paper tally sheet: section {{section}}, "{{title}}". '
    "The section is framed by black squares at its corners. Return a JSON object with these "
    "keys:\n"
    "{{fields}}\n"
    "\n"
    "How the form is filled in:\n"
    "- Numbers are handwritten one digit per box and right-aligned, so the left boxes are "
    "usually empty and the digits sit in the rightmost boxes. Read every box of a field from "
    "left to right and write only the digits you see; an empty box adds nothing. For example, "
    "boxes [ ][4][7] are 47 and [ ][ ][3] is 3.\n"
    "- Text is in block capitals, one letter per box. An empty box between letters is a space "
    "between words.\n"
    "- Circles: the filled (dark) circle is the answer. If no circle in a list is filled, use "
    "null.\n"
    "- Check each circle in a list; a filled circle is solid dark. The Day circles may all be "
    "empty, or have exactly one filled.\n"
    "\n"
    "People write some characters differently. Read them carefully:\n"
    "- A 7 may have a short bar across its middle. It is still 7, not 2 or 4.\n"
    "- A 3 may have a flat top. It is still 3, not 2.\n"
    "- A 6 may look like a C or G with a small loop at the bottom. It is still 6, not 5 or 0.\n"
    "- A 1 may have a small flag at the top. It is still 1, not 7.\n"
    "- An R may have a curly or bent leg. It is still R, not E or K.\n"
    "\n"
    "Rules:\n"
    "- Copy every number and word exactly as written; do not add up, correct or complete "
    "anything.\n"
    "- If a character is hard to read, give your best reading. Use null only when all the boxes "
    "of a field are empty.\n"
    '- A signature or stamp in the signature box means "signed".\n'
    "- Output only the JSON object."
)

# A whole cv-form page in one call. Variable: fields (every section on the page).
PAGE_PROMPT = (
    "This is a photo of a whole page of a paper tally sheet. The page has several sections. "
    "Each section is framed by black squares at its corners and headed by its letter and title, "
    'for example "A Session details". Return one JSON object with these keys, section by '
    "section:\n"
    "{{fields}}\n"
    "\n"
    "How to find each field:\n"
    "- Find the section by its heading, then the field by its label on the left of its row. In "
    "a table, the column headings above give the age group and sex of each group of boxes; read "
    "the row from left to right.\n"
    "- Each field has its own small group of boxes. Read only the boxes of that group. Never "
    "join digits from two neighbouring groups and never take a digit from the row above or "
    "below.\n"
    "\n"
    "How the form is filled in:\n"
    "- Numbers are handwritten one digit per box and right-aligned, so the left boxes are "
    "usually empty and the digits sit in the rightmost boxes. Read every box of a field from "
    "left to right and write only the digits you see; an empty box adds nothing. For example, "
    "boxes [ ][4][7] are 47 and [ ][ ][3] is 3. A single digit in the last box is still a "
    "number; do not skip it.\n"
    "- Text is in block capitals, one letter per box. Read each box once: do not repeat a "
    "letter or add one that is not there. An empty box between letters is a space between words.\n"
    "- Circles: the filled (dark) circle is the answer. If no circle in a list is filled, use "
    "null.\n"
    "- Check each circle in a list; a filled circle is solid dark. The Day circles may all be "
    "empty, or have exactly one filled.\n"
    "\n"
    "People write some characters differently. Read them carefully:\n"
    "- A 7 may have a short bar across its middle. It is still 7, not 2 or 4.\n"
    "- A 3 may have a flat top. It is still 3, not 2.\n"
    "- A 6 may look like a C or G with a small loop at the bottom. It is still 6, not 5 or 0.\n"
    "- A 1 may have a small flag at the top. It is still 1, not 7.\n"
    "- An R may have a curly or bent leg. It is still R, not E or K.\n"
    "\n"
    "Rules:\n"
    "- Copy every number and word exactly as written; do not add up, correct or complete "
    "anything.\n"
    "- If a character is hard to read, give your best reading. Use null only when all the boxes "
    "of a field are empty.\n"
    '- A signature or stamp in the signature box means "signed".\n'
    "- Output only the JSON object."
)

TAG = "formbench"
FIRST_VERSION_MESSAGE = "The prompt in code (formbench), first version"


class PromptError(SystemExit):
    """A prompt run cannot use: missing version or label, or Langfuse not reachable."""


def prompt_version(prompt: Any) -> int | None:
    """The Langfuse version of a prompt."""
    return None if prompt is None else prompt.version


def _get(langfuse: Any, name: str, choice: str | None) -> Any:
    """One prompt lookup: by version number when `choice` is digits, otherwise by label
    (production by default). Never cached, never retried, so a run sees what is in Langfuse
    right now."""
    if choice and choice.isdigit():
        return langfuse.get_prompt(name, version=int(choice), cache_ttl_seconds=0, max_retries=0)
    return langfuse.get_prompt(
        name, label=choice or "production", cache_ttl_seconds=0, max_retries=0
    )


def _unreachable(name: str, error: Exception) -> PromptError:
    return PromptError(
        f"Langfuse is required for runs and is not reachable (prompt {name}: {error}). "
        "Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST in .env."
    )


def fetch_prompt(langfuse: Any, name: str, default: str, choice: str | None = None) -> Any:
    """The prompt `name` from Langfuse. `choice` is a version number ("5"), a label
    ("production", "latest", any label) or None for production.

    The first time a prompt name is used, `default` is saved as version 1 with the production
    label. Asking for a version or label that does not exist stops: running a different
    version than the one asked for would make the results lie about what was measured.
    """
    try:
        return _get(langfuse, name, choice)
    except NotFoundError:
        pass
    except Exception as error:
        raise _unreachable(name, error) from None
    try:
        _get(langfuse, name, "latest")
        exists = True
    except NotFoundError:
        exists = False
    except Exception as error:
        raise _unreachable(name, error) from None
    if not exists and not choice:
        return langfuse.create_prompt(
            name=name,
            prompt=default,
            labels=["production"],
            tags=[TAG],
            commit_message=FIRST_VERSION_MESSAGE,
        )
    raise PromptError(
        f"Langfuse has no {name} prompt '{choice or 'production'}'. "
        f"Prompts > {name} in Langfuse lists its versions and labels."
    )


def compile_prompt(prompt: Any, variables: dict[str, str]) -> str:
    """The prompt text with its {{variables}} filled in."""
    return prompt.compile(**variables)
