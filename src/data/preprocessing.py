"""
Data preprocessing utilities for EHR datasets.
"""
from typing import List
from tqdm import tqdm


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
    
    # Get EOS token from tokenizer
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
    
    total_patients = len(base_dataset)
    print(f"  - Processing {total_patients} patients...")
    if eos_token:
        print(f"  - Adding EOS token '{eos_token}' to each sequence")
    else:
        print("  ⚠️  WARNING: Tokenizer has no EOS token!")
    
    # Use tqdm for progress bar
    for i in tqdm(range(total_patients), desc="  Extracting text", unit="patient"):
        item = base_dataset[i]
        if item is not None:
            text = item['text']
            # Remove <end> token (if still present)
            text = text.replace('<end>', '')
            
            # Clean up leading "; "
            if text.startswith('; '):
                text = text[2:]
            
            # Add EOS token at the end of each sequence
            if eos_token and not text.endswith(eos_token):
                text = text + eos_token
            
            text_list.append(text)
    
    print(f"  ✓ Extracted {len(text_list)} valid narratives (skipped {total_patients - len(text_list)} None items)")
    return text_list