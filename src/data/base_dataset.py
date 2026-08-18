from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset

from src.data.base_token_translator import BaseTokenTranslator

HEADER_SPLIT_MARKER = " <HEADER_SPLIT> "


def tokens_to_phrases(
    string_codes: List[str],
    translator: BaseTokenTranslator,
    *,
    include_header_split: bool = True,
) -> List[str]:
    """
    Convert raw EHR token strings into natural-language phrases.

    Handles:
      1. Optional HEADER_SPLIT marker before the first clinical event
      2. Measurable concept + numeric value (+ optional unit) fused into one phrase
      3. Everything else as a single-token translation

    Empty translator outputs are skipped (same as the LLM narrative path).
    """
    phrases: List[str] = []
    split_inserted = False
    i = 0

    while i < len(string_codes):
        code = str(string_codes[i])

        if include_header_split and not split_inserted and not translator.is_header_concept(code):
            phrases.append(HEADER_SPLIT_MARKER)
            split_inserted = True

        if translator.is_measurable_concept(code) and i + 1 < len(string_codes):
            next_code = str(string_codes[i + 1])
            if translator.is_numeric_value(next_code):
                concept = translator.translate(code)
                value = translator.translate(next_code)
                unit = ""
                increment = 2
                if i + 2 < len(string_codes):
                    potential_unit = str(string_codes[i + 2])
                    if not translator.is_new_event_code(potential_unit):
                        unit = translator.translate(potential_unit)
                        increment = 3
                if concept and value:
                    if unit:
                        phrases.append(
                            f"{concept.rstrip('; ').strip()}; "
                            f"{value.rstrip('; ').strip()}; "
                            f"{unit.rstrip('; ').strip()}; "
                        )
                    else:
                        phrases.append(
                            f"{concept.rstrip('; ').strip()}; "
                            f"{value.rstrip('; ').strip()}; "
                        )
                i += increment
                continue

        phrase = translator.translate(code)
        if phrase:
            phrases.append(phrase)
        i += 1

    return phrases


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
        file_fraction: Optional[float] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        '''
        Load patient records from data directory.

        ``file_fraction`` randomly samples that share of pickle shards (0–1).
        ``limit`` takes the first N shards.  When both are set, ``file_fraction``
        is applied first, then ``limit``.
        '''
        
        split_dir = os.path.join(data_dir, split)
        pkl_files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.pkl')
        )
        n_total = len(pkl_files)

        if file_fraction is not None and file_fraction < 1.0:
            n_keep = max(1, int(n_total * file_fraction))
            rng = random.Random(seed)
            pkl_files = sorted(rng.sample(pkl_files, n_keep))
            print(
                f"  Loading {n_keep}/{n_total} pickle shards "
                f"({file_fraction * 100:.0f}% of {split})",
                flush=True,
            )

        if limit is not None:
            pkl_files = pkl_files[:limit]
        
        records: List[Dict] = []
        for file_path in tqdm(pkl_files, desc=f'Loading {split}'):
            with open(file_path, 'rb') as f:
                records.extend(pickle.load(f))
        return records

    def _tokens_to_phrases(
        self,
        string_codes: List[str],
        *,
        include_header_split: bool = True,
    ) -> List[str]:
        '''
        Convert a list of raw token strings into their translated natural language phrases.

        See ``tokens_to_phrases`` for pairing rules and HEADER_SPLIT behaviour.
        '''
        return tokens_to_phrases(
            string_codes,
            self.translator,
            include_header_split=include_header_split,
        )

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

