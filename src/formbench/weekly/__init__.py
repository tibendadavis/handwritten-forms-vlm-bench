"""The weekly workflow a health facility would use. Nothing is ever sent to DHIS2.

draft.py    read this week's photos into daily values, flags and a review page
state.py    approved readings per register book: re-reads, late entries, corrections
review.py   review.html and decisions.yaml for the person approving
approve.py  approval by a named person -> datavalueset.json (written, never posted)
"""
