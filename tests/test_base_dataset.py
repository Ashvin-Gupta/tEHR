"""
Tests for BaseDataset._tokens_to_phrases().

Uses a minimal concrete subclass (TestableDataset) backed by a MockTranslator
so no real files are needed.

Covers:
- Demographic header tokens are translated before the split marker
- <HEADER_SPLIT> is inserted exactly once, before the first clinical token
- Measurement + numeric value pairs are combined into one phrase
- Measurement + numeric value + unit triplets are combined
- Measurement followed by a non-numeric token falls back to single translation
- Unknown/empty tokens are handled gracefully
- All-header sequences: no split marker inserted
- HEADER_SPLIT format compatibility check with ClassificationCollator
"""
import pytest
from typing import Dict, Any, List

from src.data.base_dataset import BaseDataset
from src.data.base_token_translator import BaseTokenTranslator


# ---------------------------------------------------------------------------
# Minimal translator for testing _tokens_to_phrases in isolation
# ---------------------------------------------------------------------------

class MockTranslator(BaseTokenTranslator):
    """Deterministic translator that maps a fixed token set."""

    _HEADER = {"<start>", "AGE", "GENDER//F", "GENDER//M", "RACE//WHITE"}
    _MEASURABLE = {"LAB//hgb", "BMI", "MEASUREMENT//sbp"}
    _NEW_EVENT = {"LAB//hgb", "DIAGNOSIS//I10", "TIME 2020", "MEASUREMENT//sbp"}

    def translate(self, token: str) -> str:
        mapping = {
            "<start>": "<start>; ",
            "AGE": "Age; ",
            "GENDER//F": "Female; ",
            "GENDER//M": "Male; ",
            "RACE//WHITE": "Race White; ",
            "LAB//hgb": "Haemoglobin; ",
            "MEASUREMENT//sbp": "Systolic BP; ",
            "BMI": "BMI; ",
            "13.5": "13.5; ",
            "95": "95; ",
            "mmol/L": "mmol/L; ",
            "DIAGNOSIS//I10": "Hypertension; ",
            "TIME 2020": "2020; ",
        }
        return mapping.get(token, f"{token}; ")

    def is_measurable_concept(self, token: str) -> bool:
        return token in self._MEASURABLE

    def is_numeric_value(self, token: str) -> bool:
        return str(token).replace(".", "", 1).isdigit()

    def is_new_event_code(self, token: str) -> bool:
        return token in self._NEW_EVENT

    def is_header_concept(self, token: str) -> bool:
        return token in self._HEADER


class TestableDataset(BaseDataset):
    """Concrete BaseDataset subclass used only for testing _tokens_to_phrases."""

    def __init__(self, translator):
        self.translator = translator

    def __len__(self):
        return 0

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {}

    # Expose _tokens_to_phrases publicly for testing
    def tokens_to_phrases(self, codes: List[str]) -> List[str]:
        return self._tokens_to_phrases(codes)


@pytest.fixture
def dataset():
    return TestableDataset(MockTranslator())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokensToPhrases:

    def test_returns_list(self, dataset):
        result = dataset.tokens_to_phrases(["<start>", "AGE", "DIAGNOSIS//I10"])
        assert isinstance(result, list)

    def test_empty_token_list(self, dataset):
        result = dataset.tokens_to_phrases([])
        assert result == []

    def test_header_split_inserted_once(self, dataset):
        tokens = ["<start>", "AGE", "GENDER//F", "DIAGNOSIS//I10", "TIME 2020"]
        phrases = dataset.tokens_to_phrases(tokens)
        text = "".join(phrases)
        assert text.count("<HEADER_SPLIT>") == 1

    def test_header_split_position(self, dataset):
        """Split marker must appear before the first clinical (non-header) token."""
        tokens = ["<start>", "GENDER//F", "DIAGNOSIS//I10"]
        phrases = dataset.tokens_to_phrases(tokens)
        text = "".join(phrases)
        # All header translations come before the split
        split_pos = text.index("<HEADER_SPLIT>")
        assert "Female" in text[:split_pos]
        assert "Hypertension" in text[split_pos:]

    def test_no_split_for_all_header_tokens(self, dataset):
        """No split marker if the entire sequence is demographic headers."""
        tokens = ["<start>", "AGE", "GENDER//F"]
        phrases = dataset.tokens_to_phrases(tokens)
        text = "".join(phrases)
        assert "<HEADER_SPLIT>" not in text

    def test_header_translations_present(self, dataset):
        tokens = ["<start>", "AGE", "GENDER//F", "DIAGNOSIS//I10"]
        text = "".join(dataset.tokens_to_phrases(tokens))
        assert "Age" in text
        assert "Female" in text

    def test_clinical_translation_present(self, dataset):
        tokens = ["<start>", "GENDER//M", "DIAGNOSIS//I10"]
        text = "".join(dataset.tokens_to_phrases(tokens))
        assert "Hypertension" in text

    # --- measurement pairing ---

    def test_measurement_value_pair(self, dataset):
        """LAB//hgb + 13.5  →  single combined phrase, not two separate ones."""
        tokens = ["<start>", "GENDER//F", "LAB//hgb", "13.5"]
        phrases = dataset.tokens_to_phrases(tokens)
        text = "".join(phrases)
        # Combined phrase must contain both concept name and value
        assert "Haemoglobin" in text
        assert "13.5" in text
        # The two tokens should be merged (no standalone "Haemoglobin; 13.5;" as separate phrases)
        haemoglobin_phrases = [p for p in phrases if "Haemoglobin" in p]
        assert len(haemoglobin_phrases) == 1
        assert "13.5" in haemoglobin_phrases[0]

    def test_measurement_value_unit_triplet(self, dataset):
        """LAB//hgb + 13.5 + mmol/L (not a new event) → single triple phrase."""
        tokens = ["<start>", "GENDER//F", "LAB//hgb", "13.5", "mmol/L"]
        phrases = dataset.tokens_to_phrases(tokens)
        combined = [p for p in phrases if "Haemoglobin" in p]
        assert len(combined) == 1
        assert "mmol/L" in combined[0]

    def test_measurement_followed_by_new_event_not_paired(self, dataset):
        """If the token after a measurable concept is another event code, do NOT pair."""
        # DIAGNOSIS//I10 is a new event code, not a numeric value
        tokens = ["<start>", "GENDER//F", "LAB//hgb", "DIAGNOSIS//I10"]
        phrases = dataset.tokens_to_phrases(tokens)
        # They should be two separate translations
        haemoglobin_phrases = [p for p in phrases if "Haemoglobin" in p]
        hypertension_phrases = [p for p in phrases if "Hypertension" in p]
        assert len(haemoglobin_phrases) == 1
        assert len(hypertension_phrases) == 1
        # They must NOT be combined
        assert "Hypertension" not in haemoglobin_phrases[0]

    def test_multiple_measurements(self, dataset):
        tokens = ["GENDER//F", "LAB//hgb", "13.5", "MEASUREMENT//sbp", "95"]
        phrases = dataset.tokens_to_phrases(tokens)
        text = "".join(phrases)
        assert "Haemoglobin" in text
        assert "13.5" in text
        assert "Systolic BP" in text
        assert "95" in text

    # ---------------------------------------------------------------------------
    # HEADER_SPLIT format compatibility with ClassificationCollator
    # ---------------------------------------------------------------------------

    def test_header_split_detectable_by_collator(self, dataset):
        """
        The ClassificationCollator looks for SPLIT_TOKEN = ' <HEADER_SPLIT> '
        (space on both sides).

        _tokens_to_phrases appends '<HEADER_SPLIT>' with no trailing space.
        The preceding phrase ends with '; ' which provides a leading space.
        But there is NO trailing space after <HEADER_SPLIT>.

        KNOWN FORMAT MISMATCH:
        - Produced text:  '...Female; <HEADER_SPLIT>Hypertension; '
        - Collator seeks: ' <HEADER_SPLIT> '  (trailing space required)

        Fix (either option):
          A) In _tokens_to_phrases:  phrases.append(' <HEADER_SPLIT> ')
          B) In ClassificationCollator:  SPLIT_TOKEN = '<HEADER_SPLIT>'

        This test documents the expected correct behaviour after the fix.
        Once fixed, the collator will correctly use smart truncation that
        always preserves the demographic header.
        """
        tokens = ["<start>", "GENDER//F", "DIAGNOSIS//I10", "TIME 2020"]
        text = "".join(dataset.tokens_to_phrases(tokens))

        # The marker must be present somewhere
        assert "<HEADER_SPLIT>" in text, (
            "No <HEADER_SPLIT> marker found at all. "
            "Check that _tokens_to_phrases inserts it correctly."
        )

        # Check what the collator actually expects
        COLLATOR_SPLIT_TOKEN = " <HEADER_SPLIT> "
        assert COLLATOR_SPLIT_TOKEN in text, (
            f"<HEADER_SPLIT> is in the text but NOT in the format the collator "
            f"expects ('{COLLATOR_SPLIT_TOKEN}').\n"
            f"Actual text: {repr(text)}\n"
            "Fix: In base_dataset.py, change\n"
            "    phrases.append('<HEADER_SPLIT>')\n"
            "to\n"
            "    phrases.append(' <HEADER_SPLIT> ')\n"
            "OR in classification_collator.py change\n"
            "    SPLIT_TOKEN = ' <HEADER_SPLIT> '\n"
            "to\n"
            "    SPLIT_TOKEN = '<HEADER_SPLIT>'"
        )
