from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

from src.data.base_token_translator import BaseTokenTranslator

class BaseDataset(ABC, Dataset):
    '''
    Shared scaffolding for all EHR dataset classes

    Subclasses must:
    - set set.translator in their __init__
    - implement __len__ and __getitem__

    '''

    translator: BaseTokenTranslator # Must be set by subclass __init__
    
    def _load_data(
        self,
        data_dir: str,
        split: str,
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        '''
        Load patient records from data directory
        '''
        
        split_dir = os.path.join(data_dir, split)
        pkl_files = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.pkl')
        ]

        if limit is not None:
            pkl_files = pkl_files[:limit]
        
        records: List[Dict] = []
        for file_path in tqdm(pkl_files, desc=f'Loading {split}'):
            with open(file_path, 'rb') as f:
                records.extend(pickle.load(f))
        return records

    def _tokens_to_phrases(self, string_codes: List[str]) -> List[str]:
        '''
        Concert a list of raw token strings into their translated natural language 

        Convert a list of raw token strings into their translated natural language phrases

        Handles three cases per position
        1. Header tokens: translated normally, no split makrer interted yet
        2. Measurable concept + numeric value (+ optional unit): combined into single phrase
        3. Everything else: single token translation

        The '<HEADER_SPLIT>' is inserted at the boundary between demographic header tokens and first clinical event.
        It is kept for pretraining.
        During classification, the ClassificationCollator uses this to split the text into header and body parts.
        '''
        phrases: List[str] = []
        split_inserted = False
        i = 0
        
        while i < len(string_codes):
            code = str(string_codes[i])

            # Insert the header/body split marker before first clinical concept
            if not split_inserted and not self.translator.is_header_concept(code):
                phrases.append(' <HEADER_SPLIT> ')
                split_inserted = True

            # Measurement pairing
            if (self.translator.is_measurable_concept(code) and i + 1 < len(string_codes)):
                next_code = str(string_codes[i+1])
                if self.translator.is_numeric_value(next_code):
                    concept = self.translator.translate(code)
                    value = self.translator.translate(next_code)
                    unit = ""
                    increment = 2
                    if i + 2 < len(string_codes):
                        potential_unit = str(string_codes[i+2])
                        if not self.translator.is_new_event_code(potential_unit):
                            unit = self.translator.translate(potential_unit)
                            increment = 3
                    if concept and value:
                        if unit:
                            phrases.append(
                                f"{concept.rstrip('; ').strip()}; {value.rstrip('; ').strip()}; {unit.rstrip('; ').strip()}; "

                            )
                        else:
                            phrases.append(f"{concept.rstrip('; ').strip()}; {value.rstrip('; ').strip()}; ")
                    i += increment
                    continue

            # Single token translation
            phrase = self.translator.translate(code)
            if phrase:
                phrases.append(phrase)
            i += 1

        return phrases

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

