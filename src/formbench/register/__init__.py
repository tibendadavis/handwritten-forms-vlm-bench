"""The immunisation register: everything specific to this form.

spec.py         register columns -> DHIS2 data elements and category option combos
prompts.py      the JSON schema and prompt sent with every page photo
reader.py       read a page or cover photo with a model
handwriting.py  parse handwritten dates and ages
aggregate.py    readings -> daily DHIS2 counts, with flags for what a person must check
"""
