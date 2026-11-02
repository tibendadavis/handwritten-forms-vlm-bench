"""How well did a model read? Readings are compared with the truth.

scoring.py   the scores, each with a plain label and meaning; cells right under the right column
truth.py     truth files for real photos: columns by key, letter or both
review.py    the benchmark's stand-in reviewer for synthetic scenarios
bench.py     run a folder (photos with truth, or a scenario) per model; compare models
photo.py     one photo, one model: every wrong cell and the scores
tracking.py  Langfuse: score definitions, datasets and experiment runs
"""
