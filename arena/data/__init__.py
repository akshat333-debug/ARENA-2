"""Public-dataset fetch + the benign-traffic profile it produces (M9).

``fetch`` is a one-shot script; the runtime only ever touches
:func:`arena.data.toucan.load_profile`, which returns ``None`` when ``data/``
is absent so the test suite never needs the datasets.
"""

from arena.data.toucan import BenignProfile, build_profile, category_of_spec, classify_name, load_profile

__all__ = [
    "BenignProfile",
    "build_profile",
    "category_of_spec",
    "classify_name",
    "load_profile",
]
