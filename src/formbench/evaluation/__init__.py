"""How well did a model read? Readings are compared with the truth.

scoring.py   page scores: text (CER, WER, ANLS), cells (field F1, STP), structure (Tree F1)
review.py    the benchmark's stand-in reviewer and the integrity scores it measures
bench.py     run a folder (scenario or photos) for one or more models; combine runs
photo.py     one photo, one model: every wrong cell and the page scores
tracking.py  Langfuse: score definitions, datasets and experiment runs
"""
