"""In-app help keys and the analyst guide must stay in sync."""

import re
from pathlib import Path

from dwells.pol.help import HELP

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "analyst_guide.md"


def test_every_help_key_has_a_guide_section():
    markers = set(re.findall(r"<!-- help:(\w+) -->", GUIDE.read_text()))
    assert set(HELP) <= markers, sorted(set(HELP) - markers)
    assert markers <= set(HELP), sorted(markers - set(HELP))
