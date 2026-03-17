from typing import Dict
import pandas as pd
from src.data.base_token_translator import BaseTokenTranslator
import logging
logger = logging.getLogger(__name__)

class MIMICTokenTranslator(BaseTokenTranslator):
    """
    Translates tokenised MIMIC-IV EHR codes to natural language.

    Lookup files required (all gzip-compressed CSV from MIMIC-IV hosp module):
      - labitems_file:   d_labitems.csv.gz   → columns 'itemid', 'label'
      - diagnoses_file:  d_icd_diagnoses.csv.gz → columns 'icd_version', 'icd_code', 'long_title'
      - procedures_file: d_icd_procedures.csv.gz → columns 'icd_version', 'icd_code', 'long_title'
    """

    def __init__(
        self,
        lab_lookup: Dict[int, str],       # itemid (int) -> label
        diag_lookup: Dict[str, str],      # "version_code" -> long_title
        proc_lookup: Dict[str, str],      # "version_code" -> long_title
    ):
        self.lab_lookup = lab_lookup
        self.diag_lookup = diag_lookup
        self.proc_lookup = proc_lookup

    @classmethod
    def from_csv_files(
        cls,
        labitems_file: str,
        diagnoses_file: str,
        procedures_file: str,
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

        return cls(lab_lookup, diag_lookup, proc_lookup)

    def translate(self, token: str) -> str:
        if not isinstance(token, str):
            return ""
        try:
            if token.startswith("LAB//"):
                parts = token.split("//")
                return f" {self.lab_lookup.get(int(parts[1]), parts[1])};"
            elif token.startswith("DIAGNOSIS//ICD//"):
                parts = token.split("//")
                return f" {self.diag_lookup.get(f'{parts[2]}_{parts[3]}', parts[3])};"
            elif token.startswith("PROCEDURE//ICD//"):
                parts = token.split("//")
                return f" {self.proc_lookup.get(f'{parts[2]}_{parts[3]}', parts[3])};"
            elif token.startswith("TIME "):
                return f" {token[5:]};"
            elif token.startswith("MEDICATION//"):
                parts = token.split("//")
                if "START" in parts:
                    return f" start {parts[-1]};"
                return f" {parts[1]} {parts[-1]};"
            elif any(x in token for x in ("ADMISSION", "DISCHARGE", "TRANSFER")):
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
            # Need to add in this translation for the DRG codes
            elif token.startswith("DRG//"):
                return f"Drug Code {token.split('//')[1]};"
            else:
                return ""
                # return f" {token.replace('//', ' ').lower()};"
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