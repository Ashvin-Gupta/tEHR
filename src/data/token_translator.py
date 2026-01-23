"""
Token translator for converting EHR token codes to natural language.
"""
import pandas as pd
from typing import Dict, Optional


class EHRTokenTranslator:
    """
    Translates EHR token codes to human-readable natural language.
    
    This class encapsulates all the logic for converting various EHR token types
    (medical codes, lab values, demographics, etc.) into readable text.
    """
    
    def __init__(
        self,
        medical_lookup: Dict[str, str],
        lab_lookup: Dict[str, str],
        region_lookup: Dict[str, str],
        time_lookup: Dict[str, str]
    ):
        """
        Initialize the translator with lookup dictionaries.
        
        Args:
            medical_lookup: Maps medical codes to descriptions.
            lab_lookup: Maps lab codes to descriptions.
            region_lookup: Maps region IDs to names.
            time_lookup: Maps time codes to descriptions.
        """
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
    ) -> 'EHRTokenTranslator':
        """
        Create a translator from CSV lookup files.
        
        Args:
            medical_lookup_file: Path to medical codes CSV.
            lab_lookup_file: Path to lab codes CSV.
            region_lookup_file: Path to region codes CSV.
            time_lookup_file: Path to time codes CSV.
        
        Returns:
            Initialized EHRTokenTranslator.
        """
        medical_df = pd.read_csv(medical_lookup_file)
        medical_lookup = pd.Series(
            medical_df['term'].values,
            index=medical_df['code'].astype(str).str.upper()
        ).to_dict()
        
        lab_df = pd.read_csv(lab_lookup_file)
        lab_lookup = pd.Series(
            lab_df['term'].values,
            index=lab_df['code'].astype(str).str.upper()
        ).to_dict()

        region_df = pd.read_csv(region_lookup_file)
        region_lookup = pd.Series(
            region_df['Description'].values,
            index=region_df['regionid'].astype(str).str.upper()
        ).to_dict()

        time_df = pd.read_csv(time_lookup_file)
        time_lookup = pd.Series(
            time_df['term'].values,
            index=time_df['code'].astype(str).str.upper()
        ).to_dict()
        
        return cls(medical_lookup, lab_lookup, region_lookup, time_lookup)
    
    def translate(self, token_string: str) -> str:
        """
        Translate a single token string to natural language.
        
        Args:
            token_string: The token code to translate.
        
        Returns:
            Translated string with trailing "; " separator.
        """
        if not isinstance(token_string, str):
            return ""
        
        try:
            # Time interval tokens
            if token_string.startswith('<time_interval_'):
                time_part = token_string.split('_')[-1].strip('>')
                return f"{self.time_lookup.get(time_part, time_part)}; "
            
            # Age tokens
            elif token_string.startswith('AGE: ') or token_string.startswith('AGE'):
                return f"{token_string}; "
            
            # BMI
            elif token_string.startswith('MEDICAL//BMI'):
                return f"{token_string.split('//')[1]}; "
            
            # Medical codes
            elif token_string.startswith('MEDICAL//'):
                code = token_string.split('//')[1].upper()
                return f"{self.medical_lookup.get(code, code.replace('_', ' ').title())}; "
            
            # Measurement codes
            elif token_string.startswith('MEASUREMENT//'):
                code = token_string.split('//')[1].upper()
                description = self.medical_lookup.get(code, code.replace('_', ' ').title())
                return f"{description}; "
            
            # Lab codes
            elif token_string.startswith('LAB//'):
                code = token_string.split('//')[1].upper()
                return f"{self.lab_lookup.get(code, code.replace('_', ' ').title())}; "
            
            # Demographics
            elif token_string.startswith(('GENDER//', 'ETHNICITY//')):
                parts = token_string.split('//')
                return f"Demographic {parts[0].title()} {parts[1].title()}; "
            
            # Region
            elif token_string.startswith('REGION//'):
                parts = token_string.split('//')
                return f"{parts[0].title()} {self.region_lookup.get(parts[1], parts[1]).title()}; "
            
            # Lifestyle
            elif token_string.startswith('LIFESTYLE//'):
                code = token_string.split('//')[1].upper()
                return f"{code.title()}; "
            
            # Numeric values
            elif token_string.replace('.', '', 1).isdigit():
                return f"{token_string}; "
            
            # Quantile values (Q1, Q2, Q3, Q4)
            elif token_string.startswith('Q') and len(token_string) <= 4 and token_string[1:].isdigit():
                return f"{token_string[1:]}; "
            
            # Lab value categories
            elif token_string.startswith(('low', 'normal', 'high', 'very low', 'very high')) and len(token_string) <= 9:
                return f"{token_string}; "
            
            # Special tokens
            elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return token_string + "; "
            
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"--- DEBUG: FAILED TO TRANSLATE TOKEN ---")
            print(f"Problematic Token String: '{token_string}'")
            print(f"Error: {e}")
            return "---ERROR_TRANSLATING---"
    
    @staticmethod
    def is_measurable_concept(token: str) -> bool:
        """Check if a token represents a measurable concept (needs a value)."""
        return token.startswith(('LAB//', 'MEASUREMENT//', 'MEDICAL//BMI', 'MEDICAL//bp_'))
    
    @staticmethod
    def is_numeric_value(token: str) -> bool:
        """Check if a token represents a numeric value."""
        safe_token = str(token)
        return safe_token.replace('.', '', 1).isdigit()
    
    @staticmethod
    def is_new_event_code(token: str) -> bool:
        """Check if a token starts a new event (not a unit or continuation)."""
        return str(token).startswith((
            'LAB//', 'MEDICAL//', 'MEASUREMENT//',
            'AGE:', 'AGE', 'GENDER//', 'ETHNICITY//', 'REGION//',
            'LIFESTYLE//', '<start>', '<end>', '<time', 'MEDS_BIRTH'
        ))
