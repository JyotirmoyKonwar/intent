from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SKILLS = FIXTURES / "skills"
GOLD = FIXTURES / "gold"
GOLD_FILES = sorted(GOLD.glob("*.json"))


@pytest.fixture
def gold_json():
    """Raw JSON text of one gold file, for tests that mutate it."""
    return (GOLD / "11-sd-destination.json").read_text()
