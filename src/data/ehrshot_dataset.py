import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any

# Map the pipeline split names (used everywhere else in the codebase) to the
# split values stored in the EHRSHOT CSV ('train', 'val', 'test').
_SPLIT_ALIASES = {
    'tuning': 'val',
    'held_out': 'test',
    'train': 'train',
    'val': 'val',
    'test': 'test',
}

class EHRSHOTDataset(Dataset):
    """
    Dataset for pre-translated EHRSHOT data.
    Reads directly from the CSV produced by the EHRSHOT extraction/formatting
    script.  The CSV is expected to have at least these columns:
        patient_id, split, label, model_text

    Pass split=None (or split='all') to load every row in the CSV regardless
    of the split column — useful for external validation.
    """
    def __init__(
        self,
        csv_filepath: str,
        split: str = 'train'
    ):
        # Load the CSV
        df = pd.read_csv(csv_filepath)

        # split=None or split='all' means "use every row" (external validation).
        if split is not None and split != 'all':
            # Normalise the requested split name so callers can use either the
            # EHRSHOT CSV names ('train'/'val'/'test') or the pipeline names
            # ('train'/'tuning'/'held_out') interchangeably.
            csv_split = _SPLIT_ALIASES.get(split, split)
            if 'split' in df.columns:
                df = df[df['split'] == csv_split]

        self.records = df.to_dict('records')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        subject_id = record['patient_id']

        # Labels in the CSV may be stored as Python booleans ('True'/'False'),
        # integer strings ('1'/'0'), or plain integers.  Convert robustly.
        raw_label = record['label']
        if isinstance(raw_label, bool):
            label = int(raw_label)
        elif isinstance(raw_label, str):
            label = 1 if raw_label.strip().lower() in ('true', '1', 'yes') else 0
        else:
            label = int(raw_label)

        # The CSV column holding the model-ready free-text narrative.
        narrative = str(record['model_text'])

        return {
            'text': narrative,
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id,
        }