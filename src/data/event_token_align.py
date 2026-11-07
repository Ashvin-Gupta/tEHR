"""
Event codes for the event-token Mistral run.

Training keeps the most recent ``max_events`` codes (``recent_events``).
The window-matching helpers below recover which events sit inside a tEHR
text truncation; the current experiment does not call them.

tEHR translates each timeline to text and keeps the suffix that fits in the
causal LM's max length (Mistral: 12000). This module recovers the events
inside that suffix, using the same phrase builder and the same tail
truncation as ``ClassificationCollator``. Times, numeric values, and units
stay in the text (they consume token budget) but are not events. An event
is one code that ``should_include_token`` would keep for XGBoost.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.data.base_dataset import HEADER_SPLIT_MARKER, tokens_to_phrases
from src.data.token_filter import should_include_token

logger = logging.getLogger(__name__)

SPLIT_TOKEN = HEADER_SPLIT_MARKER
Piece = Tuple[str, Optional[str]]  # phrase text, event code or None
Span = Tuple[int, int, Optional[str]]  # start, end, event code


def is_event_code(token: str) -> bool:
    """XGBoost / encoder event, excluding the cancer-string leakage filter."""
    if not isinstance(token, str) or not token:
        return False
    if "cancer" in token.lower():
        return False
    return should_include_token(token)


def recent_events(string_codes: Sequence[str], max_events: int) -> List[str]:
    """Most recent event codes, capped at ``max_events``. No text tokenisation."""
    events = [str(code) for code in string_codes if is_event_code(str(code))]
    limit = int(max_events)
    if limit > 0 and len(events) > limit:
        return events[-limit:]
    return events


def iter_pieces(
    string_codes: Sequence[str],
    translator: Any,
    *,
    drop_time: bool = False,
    drop_values: bool = False,
) -> List[Piece]:
    """
    Phrase pieces in ``tokens_to_phrases`` order.

    Each piece is the exact phrase text plus the source event code when that
    phrase is an included event. The header-split marker has no event.
    """
    pieces: List[Piece] = []
    split_inserted = False
    i = 0
    codes = [str(c) for c in string_codes]

    while i < len(codes):
        code = codes[i]
        if drop_time and translator.is_time_token(code):
            i += 1
            continue
        if drop_values and translator.is_value_or_unit_token(code):
            i += 1
            continue

        if not split_inserted and not translator.is_header_concept(code):
            pieces.append((SPLIT_TOKEN, None))
            split_inserted = True

        if translator.is_measurable_concept(code) and i + 1 < len(codes):
            next_code = codes[i + 1]
            if translator.is_numeric_value(next_code) and not translator.is_new_event_code(next_code):
                concept = translator.translate(code)
                increment = 2
                unit_code = None
                if i + 2 < len(codes):
                    potential = codes[i + 2]
                    if not translator.is_new_event_code(potential):
                        unit_code = potential
                        increment = 3
                value = translator.translate(next_code)
                unit = translator.translate(unit_code) if unit_code else ""
                phrase = ""
                if concept and value:
                    if unit:
                        phrase = (
                            f"{concept.rstrip('; ').strip()}; "
                            f"{value.rstrip('; ').strip()}; "
                            f"{unit.rstrip('; ').strip()}; "
                        )
                    else:
                        phrase = (
                            f"{concept.rstrip('; ').strip()}; "
                            f"{value.rstrip('; ').strip()}; "
                        )
                if phrase:
                    event = code if is_event_code(code) else None
                    pieces.append((phrase, event))
                i += increment
                continue

        phrase = translator.translate(code)
        if phrase:
            event = code if is_event_code(code) else None
            pieces.append((phrase, event))
        i += 1

    return pieces


def _join(pieces: Sequence[Piece]) -> Tuple[str, List[Span]]:
    parts: List[str] = []
    spans: List[Span] = []
    cursor = 0
    for text, event in pieces:
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor, event))
    return "".join(parts), spans


def _replace_span(text: str, spans: List[Span], old: str, new: str) -> Tuple[str, List[Span]]:
    if not old or old not in text:
        return text, spans
    mapping: List[Optional[int]] = []
    out: List[str] = []
    new_i = 0
    i = 0
    while i < len(text):
        if text.startswith(old, i):
            out.append(new)
            mapping.extend([None] * len(old))
            new_i += len(new)
            i += len(old)
        else:
            out.append(text[i])
            mapping.append(new_i)
            new_i += 1
            i += 1
    mapping.append(new_i)

    def map_pos(pos: int) -> int:
        if pos >= len(text):
            return mapping[-1]
        if mapping[pos] is not None:
            return mapping[pos]  # type: ignore[return-value]
        j = pos
        while j < len(text) and mapping[j] is None:
            j += 1
        return mapping[j] if j < len(text) else mapping[-1]  # type: ignore[return-value]

    moved = []
    for start, end, event in spans:
        ns, ne = map_pos(start), map_pos(end)
        if ne < ns:
            ne = ns
        moved.append((ns, ne, event))
    return "".join(out), moved


def _strip_span(text: str, spans: List[Span]) -> Tuple[str, List[Span]]:
    lead = len(text) - len(text.lstrip())
    trail = len(text.rstrip())
    clipped = []
    for start, end, event in spans:
        ns = min(max(start, lead), trail) - lead
        ne = min(max(end, lead), trail) - lead
        clipped.append((ns, ne, event))
    return text[lead:trail], clipped


def _tokenize(tokenizer: Any, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    if text == "":
        return [], []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = [int(x) for x in list(ids)]
    offsets = [(int(a), int(b)) for a, b in list(offsets)]
    via_encode = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(via_encode, "tolist"):
        via_encode = via_encode.tolist()
    via_encode = [int(x) for x in list(via_encode)]
    if via_encode != ids:
        # Collator truncates ``encode`` ids. Keep that decision if the fast
        # call ever disagrees, and stretch offsets so event attribution still
        # has one span per surviving id.
        ids = via_encode
        if len(offsets) != len(ids):
            step = max(1, len(text) // max(1, len(ids)))
            offsets = []
            cursor = 0
            for n in range(len(ids)):
                nxt = len(text) if n == len(ids) - 1 else min(len(text), cursor + step)
                offsets.append((cursor, nxt))
                cursor = nxt
    if len(offsets) != len(ids):
        raise RuntimeError(
            f"Token offset mismatch ({len(offsets)} offsets, {len(ids)} ids)."
        )
    return ids, offsets


def _overlapping(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    if end <= start:
        return []
    hits = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if b > start and a < end:
            hits.append(i)
    return hits


def _kept_token_indexes(
    tokenizer: Any,
    header: str,
    body: Optional[str],
    tehr_max_length: int,
) -> Tuple[set, set, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return kept header-token indexes, kept body-token indexes, and offsets."""
    bos = getattr(tokenizer, "bos_token_id", None)
    cls = getattr(tokenizer, "cls_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    n_start = 1 if (cls is not None or bos is not None) else 0
    n_end = 1 if eos is not None else 0
    max_content = int(tehr_max_length) - n_start - n_end
    if max_content <= 0:
        return set(), set(), [], []

    if body is None:
        ids, offsets = _tokenize(tokenizer, header)
        # Collator fallback: encode with specials, then keep the tail.
        # ``ids`` here are content tokens only. Specials sit outside them.
        content_budget = max_content
        if len(ids) > content_budget:
            kept = set(range(len(ids) - content_budget, len(ids)))
        else:
            kept = set(range(len(ids)))
        return kept, set(), offsets, []

    header_ids, header_offsets = _tokenize(tokenizer, header)
    body_ids, body_offsets = _tokenize(tokenizer, body)
    if len(header_ids) >= max_content:
        return set(range(max_content)), set(), header_offsets, body_offsets
    remaining = max_content - len(header_ids)
    body_start = max(0, len(body_ids) - remaining)
    return (
        set(range(len(header_ids))),
        set(range(body_start, len(body_ids))),
        header_offsets,
        body_offsets,
    )


def events_in_tehr_window(
    string_codes: Sequence[str],
    translator: Any,
    tokenizer: Any,
    tehr_max_length: int,
    *,
    drop_time: bool = False,
    drop_values: bool = False,
) -> List[str]:
    """
    Event codes whose phrase tokens survive tEHR's max-length tail truncation.

    Order is timeline order. Repeated codes stay repeated. An event is kept
    when at least one of its subword tokens is inside the window.
    """
    pieces = iter_pieces(
        string_codes,
        translator,
        drop_time=drop_time,
        drop_values=drop_values,
    )
    text, spans = _join(pieces)
    text, spans = _replace_span(text, spans, "<start>", "")
    text, spans = _replace_span(text, spans, "<end>", "")
    text, spans = _strip_span(text, spans)

    if SPLIT_TOKEN in text:
        split_at = text.find(SPLIT_TOKEN)
        header = text[:split_at]
        body: Optional[str] = text[split_at + len(SPLIT_TOKEN):]
        body_origin = split_at + len(SPLIT_TOKEN)
    else:
        header = text
        body = None
        body_origin = 0

    header_keep, body_keep, header_offsets, body_offsets = _kept_token_indexes(
        tokenizer, header, body, tehr_max_length
    )

    kept: List[str] = []
    for start, end, event in spans:
        if not event or end <= start:
            continue
        if body is None or end <= (len(header) if body is not None else 0):
            # Entirely in the header string. When there is no split, header
            # holds the whole narrative.
            region_start, region_end = start, end
            hits = _overlapping(header_offsets, region_start, region_end)
            if any(i in header_keep for i in hits):
                kept.append(event)
            continue
        if start >= body_origin:
            hits = _overlapping(body_offsets, start - body_origin, end - body_origin)
            if any(i in body_keep for i in hits):
                kept.append(event)
            continue
        # A piece should not straddle the split marker. If it does, keep the
        # event when either side survives.
        h_hits = _overlapping(header_offsets, start, min(end, len(header)))
        b_hits = _overlapping(
            body_offsets,
            max(0, start - body_origin),
            max(0, end - body_origin),
        )
        if any(i in header_keep for i in h_hits) or any(i in body_keep for i in b_hits):
            kept.append(event)
    return kept


def cap_events(events: Sequence[str], max_length: int, n_special: int = 2) -> List[str]:
    """Keep the most recent events that fit in ``max_length`` with specials."""
    budget = max(0, int(max_length) - int(n_special))
    if budget <= 0 or len(events) <= budget:
        return list(events)
    return list(events[-budget:])


def joined_phrase_text(
    string_codes: Sequence[str],
    translator: Any,
    *,
    drop_time: bool = False,
    drop_values: bool = False,
) -> str:
    """Narrative text, for checking piece text against ``tokens_to_phrases``."""
    phrases = tokens_to_phrases(
        [str(c) for c in string_codes],
        translator,
        include_header_split=True,
        drop_time=drop_time,
        drop_values=drop_values,
    )
    return "".join(phrases)


def _self_check() -> None:
    class _Tok:
        bos_token_id = 1
        eos_token_id = 2
        cls_token_id = None
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return self(text, add_special_tokens=add_special_tokens)["input_ids"]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            ids = []
            offsets = []
            i = 0
            while i < len(text):
                if text[i] == " ":
                    i += 1
                    continue
                j = i
                while j < len(text) and text[j] != " ":
                    j += 1
                ids.append(10 + len(ids))
                offsets.append((i, j))
                i = j
            return {"input_ids": ids, "offset_mapping": offsets}

    class _Tr:
        def translate(self, token: str) -> str:
            table = {
                "AGE": " Age;",
                "70": "70; ",
                "LAB//1": " LabOne;",
                "3": "3; ",
                "mg": "mg; ",
                "MEDICAL//A": " AA;",
                "MEDICAL//B": " BB;",
                "MEDICAL//C": " CC;",
                "TIME 1": " day1;",
            }
            return table.get(token, "")

        def is_measurable_concept(self, token: str) -> bool:
            return token in {"AGE", "LAB//1"}

        def is_numeric_value(self, token: str) -> bool:
            return any(c.isdigit() for c in str(token))

        def is_new_event_code(self, token: str) -> bool:
            return str(token).startswith(("LAB//", "MEDICAL//", "AGE", "TIME "))

        def is_header_concept(self, token: str) -> bool:
            if str(token).startswith("AGE"):
                return True
            try:
                float(token)
                return True
            except (TypeError, ValueError):
                return False

        def is_time_token(self, token: str) -> bool:
            return str(token).startswith("TIME ")

        def is_value_or_unit_token(self, token: str) -> bool:
            return False

    translator = _Tr()
    codes = ["AGE", "70", "LAB//1", "3", "mg", "TIME 1", "MEDICAL//A", "MEDICAL//B", "MEDICAL//C"]
    pieces = iter_pieces(codes, translator)
    assert "".join(text for text, _ in pieces) == joined_phrase_text(codes, translator)

    # Header "Age;" + "70;" is 2 tokens and one event (AGE). BOS+EOS take 2,
    # so max_length 6 leaves 4 content tokens: the header plus the last 2
    # body tokens (MEDICAL//B, MEDICAL//C). The lab phrase sits in the
    # truncated prefix, value and unit are not their own events.
    events = events_in_tehr_window(codes, translator, _Tok(), tehr_max_length=6)
    assert events == ["AGE", "MEDICAL//B", "MEDICAL//C"], events
    wide = events_in_tehr_window(codes, translator, _Tok(), tehr_max_length=12)
    assert "LAB//1" in wide and wide.count("LAB//1") == 1, wide
    assert "3" not in wide and "mg" not in wide
    assert cap_events(["a", "b", "c", "d"], max_length=4, n_special=2) == ["c", "d"]
    print("event_token_align self-check ok")
    print("  narrow:", events)
    print("  wide:", wide)


if __name__ == "__main__":
    _self_check()
