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
        split: str = 'train',
        prediction: bool = True,
        text_column: str | None = None,
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

        # Use prediction narratives by default; allow config override for
        # full-trajectory pretraining when prediction=False.
        self.prediction = prediction
        self.text_column = text_column or self._resolve_text_column(df.columns)
        self.records = df.to_dict('records')

    def _resolve_text_column(self, available_columns) -> str:
        """
        Choose which narrative column to use.

        - prediction=True  -> default to model_text (prediction-style narrative)
        - prediction=False -> prefer full trajectory columns if present
        """
        cols = set(available_columns)
        if self.prediction:
            if "model_text" not in cols:
                raise ValueError("EHRSHOT CSV is missing required 'model_text' column for prediction=True.")
            return "model_text"

        for candidate in ("full_model_text", "full_text", "trajectory_text", "model_text"):
            if candidate in cols:
                return candidate

        raise ValueError(
            "Could not find a narrative text column in EHRSHOT CSV. "
            "Tried: full_model_text, full_text, trajectory_text, model_text."
        )

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
        narrative = str(record[self.text_column])

        return {
            'text': narrative,
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id,
        }