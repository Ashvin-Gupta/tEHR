"""
Data preprocessing utilities for EHR datasets.
"""
from typing import List


def extract_text(base_dataset, tokenizer) -> List[str]:
    """
    Extracts all valid text narratives from a dataset for pretraining.
    
    This function extracts text from UnifiedEHRDataset and prepares it for SFTTrainer.
    We replace custom <start> and <end> tokens with the tokenizer's BOS and EOS tokens:
    - <start> → BOS token (e.g., <s> for Llama)
    - <end> → EOS token (e.g., </s> for Llama)
    
    With packing=True in SFTTrainer, these tokens help the model understand
    where one patient record ends and another begins.
    
    Args:
        base_dataset: The dataset to extract text from.
        tokenizer: The tokenizer to use for BOS/EOS tokens.
    
    Returns:
        List of text strings ready for training.
    """
    text_list = []
    
    # Get BOS and EOS tokens from tokenizer
    bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ""
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    
    print(f"  - Processing {len(base_dataset)} patients...")
    print(f"  - Replacing <start> with '{bos_token}' and <end> with '{eos_token}'")
    
    for i in range(len(base_dataset)):
        item = base_dataset[i]
        if item is not None:
            text = item['text']
            
            # Replace custom tokens with tokenizer's special tokens
            text = text.replace('<start>', bos_token)
            text = text.replace('<end>', eos_token)
            
            # Clean up any stray "; " at the beginning
            if text.startswith('; '):
                text = text[2:]
            
            text_list.append(text)
    
    print(f"  - Extracted {len(text_list)} valid narratives.")
    return text_list
