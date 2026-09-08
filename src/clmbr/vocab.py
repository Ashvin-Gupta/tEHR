"""Load the CLMBR-T FEMR dictionary (dictionary.msgpack)."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import msgpack


class ClmbrVocab:
    """Token ids and membership for StanfordShahLab/clmbr-t-base."""

    def __init__(self, dictionary: dict):
        self.dictionary = dictionary
        self.age_mean = float(dictionary["age_stats"]["mean"])
        self.age_std = float(dictionary["age_stats"]["std"])
        vocab = dictionary["vocab"]
        self.vocab_size = len(vocab)
        self.code_lookup: dict[str, int] = {}
        self.numeric_lookup: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
        self.string_lookup: dict[tuple[str, str], int] = {}
        self.all_codes: set[str] = set()
        for i, entry in enumerate(vocab):
            code = entry.get("code_string") or ""
            if code:
                self.all_codes.add(code)
            kind = entry.get("type")
            if kind == "code" and code:
                self.code_lookup.setdefault(code, i)
            elif kind == "numeric" and code:
                self.numeric_lookup[code].append(
                    (float(entry["val_start"]), float(entry["val_end"]), i)
                )
            elif kind == "text" and code:
                self.string_lookup[(code, str(entry.get("text_string") or ""))] = i

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ClmbrVocab":
        path = os.path.join(model_dir, "dictionary.msgpack")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"dictionary.msgpack not found in {model_dir}. "
                "Run: python -m src.clmbr.build_maps --download-model"
            )
        with open(path, "rb") as f:
            dictionary = msgpack.load(f, raw=False, strict_map_key=False)
        return cls(dictionary)

    def contains(self, code: str) -> bool:
        return code in self.all_codes

    def token_id(
        self,
        code: str,
        numeric_value: Optional[float] = None,
        text_value: Optional[str] = None,
    ) -> Optional[int]:
        """FEMR 0.2.3 FEMRTokenizer.get_feature_codes for a flat vocab.

        If this code has numeric bins and a value is present, only a matching
        bin is emitted (out-of-range values are dropped). Codes with no bins
        still emit the type=code token so SNOMED/RxNorm rollups keep a value
        attached in MEDS.
        """
        bins = self.numeric_lookup.get(code)
        if numeric_value is not None and bins:
            for start, end, idx in bins:
                if start <= numeric_value < end:
                    return idx
            return None
        if text_value:
            idx = self.string_lookup.get((code, text_value))
            if idx is not None:
                return idx
        return self.code_lookup.get(code)

    def normalize_age_seconds(self, age_seconds: float) -> float:
        return (age_seconds - self.age_mean) / max(self.age_std, 1e-6)
