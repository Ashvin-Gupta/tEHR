from typing import Dict, List
import hashlib
import random
import pandas as pd
from src.data.base_token_translator import BaseTokenTranslator
import logging

logger = logging.getLogger(__name__)

# Surface forms for lexical-grounding ablations (paper analysis).
# - text:             normal clinical titles (default)
# - raw:              opaque EHR token strings for clinical concepts
# - scramble_hash:    stable nonsense IDs (Concept_<hash>)
# - scramble_shuffle: real titles permuted across codes (same category)
SURFACE_FORMS = ("text", "raw", "scramble_hash", "scramble_shuffle")


class MIMICTokenTranslator(BaseTokenTranslator):
    """
    Translates tokenised MIMIC-IV EHR codes to natural language.

    Lookup files required (all gzip-compressed CSV from MIMIC-IV hosp module):
      - labitems_file:   d_labitems.csv.gz   → columns 'itemid', 'label'
      - diagnoses_file:  d_icd_diagnoses.csv.gz → columns 'icd_version', 'icd_code', 'long_title'
      - procedures_file: d_icd_procedures.csv.gz → columns 'icd_version', 'icd_code', 'long_title'

    ``surface_form`` controls whether clinical concept *names* are lexical text,
    raw codes, or scrambled. Demographics, times, and numeric values always use
    the normal text path (only lab/ICD/med/DRG concept labels are altered).
    """

    def __init__(
        self,
        lab_lookup: Dict[int, str],       # itemid (int) -> label
        diag_lookup: Dict[str, str],      # "version_code" -> long_title
        proc_lookup: Dict[str, str],      # "version_code" -> long_title
        surface_form: str = "text",
        scramble_seed: int = 42,
    ):
        if surface_form not in SURFACE_FORMS:
            raise ValueError(
                f"Unknown surface_form={surface_form!r}. "
                f"Expected one of {SURFACE_FORMS}"
            )
        self.lab_lookup = lab_lookup
        self.diag_lookup = diag_lookup
        self.proc_lookup = proc_lookup
        self.surface_form = surface_form
        self.scramble_seed = int(scramble_seed)
        self._scramble_map: Dict[str, str] = {}
        if surface_form in ("scramble_hash", "scramble_shuffle"):
            self._build_scramble_maps()

    @classmethod
    def from_csv_files(
        cls,
        labitems_file: str,
        diagnoses_file: str,
        procedures_file: str,
        surface_form: str = "text",
        scramble_seed: int = 42,
    ) -> "MIMICTokenTranslator":
        lab_lookup = (
            pd.read_csv(labitems_file)
            .set_index("itemid")["label"]
            .to_dict()
        )

        diag_df = pd.read_csv(diagnoses_file)
        diag_df["key"] = diag_df["icd_version"].astype(str) + "_" + diag_df["icd_code"].astype(str)
        diag_lookup = diag_df.set_index("key")["long_title"].to_dict()

        proc_df = pd.read_csv(procedures_file)
        proc_df["key"] = proc_df["icd_version"].astype(str) + "_" + proc_df["icd_code"].astype(str)
        proc_lookup = proc_df.set_index("key")["long_title"].to_dict()

        return cls(
            lab_lookup,
            diag_lookup,
            proc_lookup,
            surface_form=surface_form,
            scramble_seed=scramble_seed,
        )

    def _build_scramble_maps(self) -> None:
        """Build key → surface string for labs / diagnoses / procedures."""
        rng = random.Random(self.scramble_seed)
        categories: Dict[str, List[tuple]] = {
            "lab": [(f"lab:{k}", str(v)) for k, v in self.lab_lookup.items()],
            "diag": [(f"diag:{k}", str(v)) for k, v in self.diag_lookup.items()],
            "proc": [(f"proc:{k}", str(v)) for k, v in self.proc_lookup.items()],
        }
        for _cat, pairs in categories.items():
            keys = [k for k, _ in pairs]
            titles = [t for _, t in pairs]
            if self.surface_form == "scramble_hash":
                for key, title in pairs:
                    digest = hashlib.sha1(
                        f"{self.scramble_seed}:{key}:{title}".encode("utf-8")
                    ).hexdigest()[:8]
                    self._scramble_map[key] = f"Concept_{digest}"
            else:  # scramble_shuffle — permute titles within category
                shuffled = titles[:]
                rng.shuffle(shuffled)
                for key, new_title in zip(keys, shuffled):
                    self._scramble_map[key] = new_title

    @staticmethod
    def _format_phrase(label: str) -> str:
        return f" {label};"

    def _alter_clinical(
        self,
        *,
        map_key: str,
        text_label: str,
        raw_token: str,
    ) -> str:
        """Apply surface_form to a clinical concept label (not values/demos)."""
        if self.surface_form == "text":
            return self._format_phrase(text_label)
        if self.surface_form == "raw":
            # Keep the opaque EHR token string (strip trailing junk).
            return self._format_phrase(raw_token.strip())
        # scramble_*
        scrambled = self._scramble_map.get(map_key)
        if scrambled is None:
            # Unseen meds / fallback keys: hash on the fly.
            digest = hashlib.sha1(
                f"{self.scramble_seed}:{map_key}:{text_label}".encode("utf-8")
            ).hexdigest()[:8]
            scrambled = f"Concept_{digest}"
            self._scramble_map[map_key] = scrambled
        return self._format_phrase(scrambled)

    def translate(self, token: str) -> str:
        if not isinstance(token, str):
            return ""
        try:
            if token.startswith("LAB//"):
                parts = token.split("//")
                itemid = int(parts[1])
                label = self.lab_lookup.get(itemid, parts[1])
                return self._alter_clinical(
                    map_key=f"lab:{itemid}",
                    text_label=str(label),
                    raw_token=token,
                )
            elif token.startswith("DIAGNOSIS//ICD//"):
                parts = token.split("//")
                key = f"{parts[2]}_{parts[3]}"
                label = self.diag_lookup.get(key, parts[3])
                return self._alter_clinical(
                    map_key=f"diag:{key}",
                    text_label=str(label),
                    raw_token=token,
                )
            elif token.startswith("PROCEDURE//ICD//"):
                parts = token.split("//")
                key = f"{parts[2]}_{parts[3]}"
                label = self.proc_lookup.get(key, parts[3])
                return self._alter_clinical(
                    map_key=f"proc:{key}",
                    text_label=str(label),
                    raw_token=token,
                )
            elif token.startswith("TIME "):
                return f" {token[5:]};"
            elif token.startswith("MEDICATION//"):
                parts = token.split("//")
                is_start = "START" in parts
                drug = parts[-1] if is_start else parts[1]
                rest = None if is_start else parts[-1]
                if self.surface_form == "text":
                    if is_start:
                        return f" start {drug};"
                    return f" {drug} {rest};"
                if self.surface_form == "raw":
                    return self._format_phrase(token)
                altered = self._alter_clinical(
                    map_key=f"med:{str(drug).lower()}",
                    text_label=str(drug),
                    raw_token=str(drug),
                ).strip().rstrip(";")
                if is_start:
                    return f" start {altered};"
                return f" {altered} {rest};"
            elif any(x in token for x in ("ADMISSION", "DISCHARGE", "TRANSFER")):
                # Keep as lightly normalised text (not a lexicon title).
                return f" {token.replace('//', ', ').replace('_', ' ').lower()};"
            elif token.startswith("GENDER//"):
                val = token.split("//")[1]
                return " Female;" if val == "F" else " Male;"
            elif token.startswith("RACE//"):
                val = token.split("//")[1].replace("RACE_", "").replace("_", " ").title()
                return f" Race {val};"
            elif token.startswith("MARITAL_STATUS//"):
                return f" {token.split('//')[1].title()};"
            elif token.startswith("AGE"):
                return " Age;"
            elif token.startswith("BMI"):
                return " BMI;"
            elif token.startswith("Blood Pressure"):
                return " Blood Pressure;"
            elif token == "<start>":
                return "<start>;"
            elif token.replace(".", "", 1).isdigit() or "/" in token:
                return f"{token}; "
            elif token.startswith("DRG//"):
                code = token.split("//")[1]
                return self._alter_clinical(
                    map_key=f"drg:{code}",
                    text_label=f"Drug Code {code}",
                    raw_token=token,
                )
            else:
                return ""
        except Exception as e:
            logger.warning("Failed to translate MIMIC token '%s': %s", token, e)
            return "---ERROR_TRANSLATING---; "

    def is_measurable_concept(self, token: str) -> bool:
        return token.startswith(("LAB//", "AGE", "BMI", "Blood Pressure"))

    @staticmethod
    def is_numeric_value(token: str) -> bool:
        # MIMIC values are permissive: just check any digit is present
        return any(c.isdigit() for c in str(token))

    def is_new_event_code(self, token: str) -> bool:
        return str(token).startswith((
            "LAB//", "DIAGNOSIS//", "PROCEDURE//",
            "TIME ",           # note the space — MIMIC uses "TIME 2020-01-15" format
            "MEDICATION//",
            "AGE", "BMI", "Blood Pressure",
            "GENDER//", "RACE//", "MARITAL_STATUS//",
            "ADMISSION", "DISCHARGE", "TRANSFER",
            "<start>",
        ))

    def is_header_concept(self, token: str) -> bool:
        t = str(token).upper()
        if t.startswith(("<START>", "AGE", "GENDER", "RACE", "MARITAL_STATUS", "BMI")):
            return True
        # Bare numeric values (age/BMI values) belong to the demographic header
        try:
            float(str(token))
            return True
        except (ValueError, TypeError):
            return False
