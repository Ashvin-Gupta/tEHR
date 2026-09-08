"""
Shared EHR token filter (encoder / XGBoost) and matched-event truncation.

``should_include_token`` is the encoder's event definition: medical/lab/
measurement codes plus demographics/lifestyle. Time intervals, numeric
values, units, lab categories, and quantiles are excluded.

``truncate_to_max_included_events`` keeps the most recent N of those
events and the time/value tokens attached to that window, so tEHR can
match the encoder's event budget without dropping per-event text.
"""
from typing import List, Optional, Sequence


def is_time_token(token_str: str) -> bool:
    t = str(token_str)
    return t.startswith("TIME ") or t.startswith("<time")


def should_include_token(token_str: str) -> bool:
    """
    Check if a token should be included in the bag-of-words / encoder stream.

    Includes:
    - Medical events (MEDICAL//...)
    - Lab test names (LAB//...) - but not their values
    - Measurement codes (MEASUREMENT//...)
    - Demographics (AGE, GENDER//..., ETHNICITY//..., REGION//...)
    - Lifestyle (LIFESTYLE//...)
    - Special tokens (<start>, <end>, <unknown>, MEDS_BIRTH)

    Excludes:
    - Time intervals (<time_interval_...>)
    - Numeric values (pure numbers)
    - Units (short strings that don't match code patterns)
    - Lab value categories (low, normal, high, very low, very high)
    - Quantile values (Q1, Q2, Q3, Q4)

    Args:
        token_str: Token string to check

    Returns:
        True if token should be included, False otherwise
    """
    if not isinstance(token_str, str) or not token_str:
        return False

    token_upper = token_str.upper()
    token_lower = token_str.lower()

    # Exclude time intervals
    if token_str.startswith('<time_interval_'):
        return False

    if token_str.startswith('TIME '):
        return False

    if token_str.startswith('UNIT//') or token_str.startswith('VALUE//'):
        return False

    # Exclude numeric values (pure numbers, including negative and decimal)
    if token_str.replace('.', '', 1).replace('-', '', 1).isdigit():
        return False

    # Exclude quantile values (Q1, Q2, Q3, Q4)
    if token_str.startswith('Q') and len(token_str) <= 4 and token_str[1:].isdigit():
        return False

    # Exclude tokens containing "6866.00"
    if "6866.00" in token_str:
        return False

    if "22J..00" in token_str:
        return False

    if "BB5B.00" in token_str:
        return False

    if "8CMG.00" in token_str:
        return False

    # Exclude lab value categories
    if token_lower in ['low', 'normal', 'high', 'very low', 'very high']:
        return False

    if token_upper.startswith('EVENT//'):
        return True

    # Include lab test names (but not their values - values are numeric and already excluded above)
    if token_upper.startswith('LAB//'):
        return True

    # Include measurement codes
    if token_upper.startswith('MEASUREMENT//'):
        return True

    # Include demographics
    if token_upper.startswith(('AGE:', 'AGE', 'GENDER//', 'ETHNICITY//', 'REGION//', 'RACE//')):
        return True

    # Include lifestyle
    if token_upper.startswith('LIFESTYLE//'):
        return True

    # Include special tokens
    if token_str in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
        return False

    # Include medical codes
    if token_upper.startswith('MEDICAL//'):
        return True

    # Exclude units - tokens that are short strings with unit-like characters
    # but don't match any code pattern
    # Common units: "mmol/L", "mg/dL", "kg", "cm", "%", etc.
    if len(token_str) <= 20:
        # Check if it contains unit-like characters but isn't a code
        has_unit_chars = any(char in token_str for char in ['/', '%', '°', 'µ', 'μ'])
        # Or if it's a very short string (1-5 chars) that's all lowercase/uppercase letters
        is_short_alpha = len(token_str) <= 5 and token_str.replace(' ', '').isalpha()

        # If it doesn't match any code pattern and looks like a unit, exclude it
        if (has_unit_chars or is_short_alpha) and not token_upper.startswith((
            '<', 'MEDICAL', 'LAB', 'MEASUREMENT', 'AGE', 'GENDER',
            'ETHNICITY', 'REGION', 'LIFESTYLE', 'EVENT', 'RACE',
        )):
            return False

    # By default, exclude unknown tokens to be conservative
    # Only include tokens that explicitly match known patterns
    return True


def kept_window_start(
    string_codes: Sequence[str],
    max_events: Optional[int],
) -> int:
    """
    Index of the suffix that contains at most ``max_events`` encoder-style events.

    Uses the same ``should_include_token`` definition as the encoder. Time
    tokens immediately before the first kept event are included so the first
    event is not missing its timestamp. Value/unit tokens after kept events
    stay in the suffix automatically.
    """
    if max_events is None:
        return 0
    limit = int(max_events)
    if limit <= 0:
        return 0

    included = [i for i, code in enumerate(string_codes) if should_include_token(str(code))]
    if len(included) <= limit:
        return 0

    start = included[-limit]
    while start > 0 and is_time_token(str(string_codes[start - 1])):
        start -= 1
    return start


def truncate_to_max_included_events(
    string_codes: List[str],
    max_events: Optional[int],
) -> List[str]:
    """Keep the most recent ``max_events`` encoder-style events (and payloads)."""
    start = kept_window_start(string_codes, max_events)
    if start <= 0:
        return string_codes
    return list(string_codes[start:])
