"""
Data preprocessing utilities for EHR datasets.
"""
from typing import List


def extract_text(base_dataset, tokenizer) -> List[str]:
    """
    Extracts all valid text narratives from a dataset for pretraining.
    
    This function extracts text from UnifiedEHRDataset and prepares it for SFTTrainer.
    We keep both <start> and <end> tokens as they mark sequence boundaries:
    - <start> marks the beginning of a patient record
    - <end> marks the end of a patient record (the model should learn to predict this)
    
    With packing=True in SFTTrainer, these tokens help the model understand
    where one patient record ends and another begins.
    
    Args:
        base_dataset: The dataset to extract text from.
        tokenizer: The tokenizer (unused but kept for consistency).
    
    Returns:
        List of text strings ready for training.
    """
    text_list = []
    
    print(f"  - Processing {len(base_dataset)} patients...")
    for i in range(len(base_dataset)):
        item = base_dataset[i]
        if item is not None:
            text = item['text']
            # Keep both <start> and <end> tokens - they're important for learning sequence boundaries
            # Clean up any stray "; " at the beginning (shouldn't happen if <start> is present)
            if text.startswith('; '):
                text = text[2:]
            text_list.append(text)
    
    print(f"  - Extracted {len(text_list)} valid narratives.")
    return text_list