"""Build packed CLMBR sequences and extract last-token representations."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch

from src.clmbr.mapping import CodeMapper
from src.clmbr.model import causal_local_mask, causal_local_mask_padded, load_clmbr_transformer
from src.clmbr.vocab import ClmbrVocab

ATTENTION_WIDTH = 496


def parse_numeric(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_time(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").astype(datetime)
    return None


BIRTH_EVENT_CODES = {"MEDS_BIRTH", "SNOMED/184099003"}


class PatientTimeline:
    __slots__ = ("birth", "birth_years", "events")

    def __init__(self):
        self.birth: Optional[datetime] = None
        self.birth_years: Optional[float] = None
        self.events: list[tuple[datetime, int, float]] = []  # time, token, age_days


def accumulate_row(
    timeline: PatientTimeline,
    code: str,
    time,
    numeric_value,
    mapper: CodeMapper,
    vocab: ClmbrVocab,
) -> None:
    if code in BIRTH_EVENT_CODES:
        ts = parse_time(time)
        if ts is not None:
            timeline.birth = ts
        return
    if code == "AGE":
        years = parse_numeric(numeric_value)
        if years is not None and timeline.birth_years is None:
            timeline.birth_years = years
        return
    mapped = mapper.map_code(code)
    if mapped.stage != "in_vocab" or not mapped.clmbr_code:
        return
    ts = parse_time(time)
    if ts is None:
        return
    token = vocab.token_id(mapped.clmbr_code, parse_numeric(numeric_value))
    if token is None:
        return
    timeline.events.append((ts, token, 0.0))


def finalize_timeline(timeline: PatientTimeline, vocab: ClmbrVocab) -> PatientTimeline:
    if not timeline.events:
        return timeline
    timeline.events.sort(key=lambda x: x[0])
    first = timeline.events[0][0]
    if timeline.birth is not None:
        birth = timeline.birth
    elif timeline.birth_years is not None:
        birth = first - timedelta(days=float(timeline.birth_years) * 365.25)
    else:
        birth = first
    timeline.birth = birth
    # FEMR 0.2.3 rotary uses age in days, and drops duplicate codes on the
    # same calendar day (codes_seen_today in BatchCreator.add_patient).
    filled = []
    seen_day = None
    seen_tokens: set[int] = set()
    for ts, token, _ in timeline.events:
        day = ts.date()
        if day != seen_day:
            seen_day = day
            seen_tokens = set()
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        age_days = (ts - birth).total_seconds() / 86400.0
        if age_days < 0:
            age_days = 0.0
        filled.append((ts, token, age_days))
    timeline.events = filled
    return timeline


def events_before(
    timeline: PatientTimeline, pred_time: datetime, max_len: int = ATTENTION_WIDTH
) -> list[tuple[datetime, int, float]]:
    kept = [e for e in timeline.events if e[0] <= pred_time]
    if len(kept) > max_len:
        kept = kept[-max_len:]
    return kept


@torch.no_grad()
def embed_sequences(
    sequences: list[list[tuple[datetime, int, float]]],
    model,
    vocab: ClmbrVocab,
    device: torch.device,
    hidden_size: int = 768,
) -> np.ndarray:
    """One representation per sequence: last token, or zeros if empty."""
    out = np.zeros((len(sequences), hidden_size), dtype=np.float32)
    nonempty = [(i, seq) for i, seq in enumerate(sequences) if seq]
    if not nonempty:
        return out
    # Process one sequence at a time (L<=496). Simple and matches local attn.
    for i, seq in nonempty:
        tokens = torch.tensor([t for _, t, _ in seq], dtype=torch.long, device=device)
        ages = torch.tensor([a for _, _, a in seq], dtype=torch.float32, device=device)
        mask = causal_local_mask(len(seq), ATTENTION_WIDTH, device)
        hidden = model(tokens, ages, mask)
        out[i] = hidden[-1].detach().float().cpu().numpy()
    return out


def sequences_to_arrays(
    sequences: list[list[tuple[datetime, int, float]]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    tokens, ages = [], []
    for seq in sequences:
        if seq:
            tokens.append(np.array([t for _, t, _ in seq], dtype=np.int32))
            ages.append(np.array([a for _, _, a in seq], dtype=np.float32))
        else:
            tokens.append(np.zeros(1, dtype=np.int32))
            ages.append(np.zeros(1, dtype=np.float32))
    return tokens, ages


def collate_clmbr_batch(rows, device: torch.device, window: int = ATTENTION_WIDTH):
    """Pad token/age rows and build a causal local mask."""
    lengths = torch.tensor([len(r[0]) for r in rows], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    batch = len(rows)
    tokens = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    ages = torch.zeros((batch, max_len), dtype=torch.float32, device=device)
    labels = torch.zeros(batch, dtype=torch.float32, device=device)
    for i, (tok, age, y) in enumerate(rows):
        n = len(tok)
        tokens[i, :n] = torch.as_tensor(tok, dtype=torch.long, device=device)
        ages[i, :n] = torch.as_tensor(age, dtype=torch.float32, device=device)
        labels[i] = float(y)
    mask = causal_local_mask_padded(lengths, max_len, window, device)
    return tokens, ages, mask, lengths, labels


def load_model(model_dir: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_clmbr_transformer(model_dir, device), device
