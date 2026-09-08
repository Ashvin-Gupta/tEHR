"""OMOP / EHRSHOT token → natural-language phrases (lexical ablations included)."""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

import pandas as pd

from src.data.base_token_translator import BaseTokenTranslator
from src.data.mimic_translator import SURFACE_FORMS

# Tables whose titles are scrambled in hash / shuffle ablations.
_CLINICAL_TABLES = {
    "measurement",
    "observation",
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
    "device_exposure",
    "note",
    "visit_occurrence",
    "visit_detail",
    "death",
}


class EHRSHOTTokenTranslator(BaseTokenTranslator):
    """Translate ``EVENT//{table}//{vocab}/{code}`` using Athena ``concept.csv``."""

    def __init__(
        self,
        code_to_name: Dict[str, str],
        surface_form: str = "text",
        scramble_seed: int = 42,
    ):
        if surface_form not in SURFACE_FORMS:
            raise ValueError(
                f"Unknown surface_form={surface_form!r}. Expected {SURFACE_FORMS}"
            )
        self.code_to_name = code_to_name
        self.surface_form = surface_form
        self.scramble_seed = int(scramble_seed)
        self._scramble_map: Dict[str, str] = {}
        if surface_form in ("scramble_hash", "scramble_shuffle"):
            self._build_scramble_maps()

    @classmethod
    def from_concept_csv(
        cls,
        concept_csv: str,
        surface_form: str = "text",
        scramble_seed: int = 42,
    ) -> "EHRSHOTTokenTranslator":
        df = pd.read_csv(
            concept_csv,
            usecols=["concept_name", "vocabulary_id", "concept_code"],
            low_memory=False,
        )
        df["code"] = (
            df["vocabulary_id"].astype(str) + "/" + df["concept_code"].astype(str)
        )
        df = df[df["concept_name"].astype(str) != df["code"]]
        lookup = dict(zip(df["code"], df["concept_name"].astype(str)))
        return cls(lookup, surface_form=surface_form, scramble_seed=scramble_seed)

    @classmethod
    def from_csv_files(
        cls,
        concept_csv: str,
        surface_form: str = "text",
        scramble_seed: int = 42,
    ) -> "EHRSHOTTokenTranslator":
        return cls.from_concept_csv(
            concept_csv, surface_form=surface_form, scramble_seed=scramble_seed
        )

    def _build_scramble_maps(self) -> None:
        rng = random.Random(self.scramble_seed)
        pairs: List[Tuple[str, str]] = [
            (code, name) for code, name in self.code_to_name.items()
        ]
        if self.surface_form == "scramble_hash":
            for code, name in pairs:
                digest = hashlib.sha1(
                    f"{self.scramble_seed}:{code}:{name}".encode("utf-8")
                ).hexdigest()[:8]
                self._scramble_map[code] = f"Concept_{digest}"
            return
        keys = [c for c, _ in pairs]
        titles = [n for _, n in pairs]
        shuffled = titles[:]
        rng.shuffle(shuffled)
        for key, title in zip(keys, shuffled):
            self._scramble_map[key] = title

    @staticmethod
    def _format_phrase(label: str) -> str:
        return f" {label};"

    def _alter_clinical(self, map_key: str, text_label: str) -> str:
        if self.surface_form == "text":
            return self._format_phrase(text_label)
        scrambled = self._scramble_map.get(map_key)
        if scrambled is None:
            digest = hashlib.sha1(
                f"{self.scramble_seed}:{map_key}:{text_label}".encode("utf-8")
            ).hexdigest()[:8]
            scrambled = f"Concept_{digest}"
            self._scramble_map[map_key] = scrambled
        return self._format_phrase(scrambled)

    def _event_name(self, code: str) -> str:
        name = self.code_to_name.get(code)
        if name and name != code:
            return str(name)
        return code.replace("/", " ")

    def translate(self, token: str) -> str:
        if not isinstance(token, str) or not token:
            return ""
        if token.startswith("EVENT//"):
            parts = token.split("//")
            table = parts[1] if len(parts) > 1 else ""
            code = parts[2] if len(parts) > 2 else token
            label = self._event_name(code)
            if table in _CLINICAL_TABLES:
                return self._alter_clinical(code, label)
            return self._format_phrase(label)
        if token.startswith("TIME "):
            return f" {token[5:]};"
        if token.startswith("GENDER//"):
            val = token.split("//", 1)[1]
            if val in {"F", "FEMALE"}:
                return " Female;"
            if val in {"M", "MALE"}:
                return " Male;"
            return f" Gender {val.title()};"
        if token.startswith("RACE//"):
            val = token.split("//", 1)[1].replace("_", " ").title()
            return f" Race {val};"
        if token.startswith("ETHNICITY//"):
            val = token.split("//", 1)[1].replace("_", " ").title()
            return f" Ethnicity {val};"
        if token.startswith("AGE"):
            return " Age;"
        if token.startswith("UNIT//"):
            return f"{token.split('//', 1)[1]}; "
        if token.startswith("VALUE//"):
            return f"{token.split('//', 1)[1]}; "
        if token == "<start>":
            return "<start>;"
        if token.replace(".", "", 1).replace("-", "", 1).isdigit():
            return f"{token}; "
        return ""

    def is_measurable_concept(self, token: str) -> bool:
        if not str(token).startswith("EVENT//"):
            return False
        table = str(token).split("//")[1] if str(token).count("//") >= 2 else ""
        return table in {"measurement", "observation"}

    @staticmethod
    def is_numeric_value(token: str) -> bool:
        t = str(token)
        if t.startswith("VALUE//"):
            return True
        return t.replace(".", "", 1).replace("-", "", 1).isdigit()

    def is_new_event_code(self, token: str) -> bool:
        return str(token).startswith((
            "EVENT//",
            "TIME ",
            "AGE",
            "GENDER//",
            "RACE//",
            "ETHNICITY//",
            "<start>",
        ))

    def is_header_concept(self, token: str) -> bool:
        t = str(token).upper()
        if t.startswith(("<START>", "AGE", "GENDER", "RACE", "ETHNICITY")):
            return True
        try:
            float(str(token))
            return True
        except (ValueError, TypeError):
            return False

    def is_value_or_unit_token(self, token_string: str) -> bool:
        t = str(token_string)
        if t.startswith(("UNIT//", "VALUE//")):
            return True
        return super().is_value_or_unit_token(token_string)
