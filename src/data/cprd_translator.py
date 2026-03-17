from typing import Dict
import pandas as pd
from src.data.base_token_translator import BaseTokenTranslator
import logging
logger = logging.getLogger(__name__)

class CPRDTokenTranslator(BaseTokenTranslator):
    '''
    Tsnalated tokensised CPRD EHR codes into natural language

    Lookup files required:
    - medical_lookup_file: CSV with columns 'code' and 'term'
    - lab_lookup_file: CSV with columns 'code' and 'term'
    - region_lookup_file: CSV with columns 'regionid' and 'Description'
    - time_lookup_file: CSV with columns 'code' and 'term'
    '''

    def __init__(
        self, 
        medical_lookup: Dict[str, str],
        lab_lookup: Dict[str, str],
        region_lookup: Dict[str, str],
        time_lookup: Dict[str, str]
    ):
        self.medical_lookup = medical_lookup
        self.lab_lookup = lab_lookup
        self.region_lookup = region_lookup
        self.time_lookup = time_lookup

    @classmethod
    def from_csv_files(
        cls,
        medical_lookup_file: str,
        lab_lookup_file: str,
        region_lookup_file: str,
        time_lookup_file: str
    ) -> 'CPRDTokenTranslator':

        medical_df = pd.read_csv(medical_lookup_file)
        medical_lookup = pd.Series(medical_df['term'].values, index=medical_df['code'].astype(str).str.upper()).to_dict()

        lab_df = pd.read_csv(lab_lookup_file)
        lab_lookup = pd.Series(lab_df['term'].values, index=lab_df['code'].astype(str).str.upper()).to_dict()

        region_df = pd.read_csv(region_lookup_file)
        region_lookup = pd.Series(region_df['Description'].values, index=region_df['regionid'].astype(str).str.upper()).to_dict()

        time_df = pd.read_csv(time_lookup_file)
        time_lookup = pd.Series(time_df['term'].values, index=time_df['code'].astype(str).str.upper()).to_dict()

        return cls(medical_lookup, lab_lookup, region_lookup, time_lookup)
    
    def translate(self, token_string: str) -> str:
        if not isinstance(token_string, str):
            return ""
        try:
            if token_string.startswith("<time_interval_"):
                time_part = token_string.split("_")[-1].strip(">")
                return f"{self.time_lookup.get(time_part, time_part)}; "
            elif token_string.startswith("AGE"):
                return f"{token_string}; "
            elif token_string.startswith("MEDICAL//BMI"):
                return f"{token_string.split('//')[1]}; "
            elif token_string.startswith("MEDICAL//"):
                code = token_string.split("//")[1].upper()
                return f"{self.medical_lookup.get(code, code.replace('_', ' ').title())}; "
            elif token_string.startswith("MEASUREMENT//"):
                code = token_string.split("//")[1].upper()
                return f"{self.medical_lookup.get(code, code.replace('_', ' ').title())}; "
            elif token_string.startswith("LAB//"):
                code = token_string.split("//")[1].upper()
                return f"{self.lab_lookup.get(code, code.replace('_', ' ').title())}; "
            elif token_string.startswith(("GENDER//", "ETHNICITY//")):
                parts = token_string.split("//")
                return f"Demographic {parts[0].title()} {parts[1].title()}; "
            elif token_string.startswith("REGION//"):
                parts = token_string.split("//")
                return f"{parts[0].title()} {self.region_lookup.get(parts[1], parts[1]).title()}; "
            elif token_string.startswith("LIFESTYLE//"):
                code = token_string.split("//")[1].upper()
                return f"{code.title()}; "
            elif token_string.replace(".", "", 1).isdigit():
                return f"{token_string}; "
            elif token_string.startswith("Q") and len(token_string) <= 4 and token_string[1:].isdigit():
                return f"{token_string[1:]}; "
            elif token_string in ("<start>", "<end>", "<unknown>", "MEDS_BIRTH"):
                return f"{token_string}; "
            else:
                return "Unknown; "
        except Exception as e:
            logger.warning("Failed to translate CPRD token '%s': %s", token_string, e)
            return "---ERROR_TRANSLATING---; "
    
    def is_measurable_concept(self, token: str) -> bool:
        return token.startswith(("LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"))

    def is_new_event_code(self, token: str) -> bool:
        return str(token).startswith((
            "LAB//", "MEDICAL//", "MEASUREMENT//",
            "AGE:", "AGE", "GENDER//", "ETHNICITY//", "REGION//",
            "LIFESTYLE//", "<start>", "<end>", "<time", "MEDS_BIRTH",
        ))

    def is_header_concept(self, token: str) -> bool:
        t = str(token).upper()
        if t.startswith(("<START>", "AGE", "GENDER", "ETHNICITY", "REGION", "MEDS_BIRTH", "LIFESTYLE")):
            return True
        if t.replace(".", "", 1).isdigit():
            return True
        if "BMI" in t:
            return True
        return False
    
    @staticmethod
    def is_numeric_value(token: str) -> bool:
        # CPRD values are permissive: just check any digit is present
        return any(c.isdigit() for c in str(token))