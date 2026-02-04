# src/data/classification_collator.py

"""
Data collator for LLM-based classification tasks.

Handles tokenization, padding, attention masks, and binary label conversion
for EHR text classification.
"""

import torch
from typing import List, Dict, Any
import warnings


class ClassificationCollator:
    """
    Collate function for LLM classification tasks.
    
    Takes raw text from UnifiedEHRDataset and tokenizes it for batch processing.
    Also converts multi-class labels to binary (cancer vs control).
    
    Args:
        tokenizer: HuggingFace tokenizer (should be the extended tokenizer from pretraining)
        max_length: Maximum sequence length for truncation
        binary_classification: If True, converts labels > 0 to 1 (cancer vs control)
    """
    
    def __init__(self, tokenizer, max_length: int = 2048, binary_classification: bool = True, truncation: bool = False,
                 handle_long_sequences: str = 'warn'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.binary_classification = binary_classification
        self.truncation = truncation
        self.handle_long_sequences = handle_long_sequences
        self._warned_once = False  # Only warn once to avoid spam
        self._long_sequence_count = 0
        self._total_sequences = 0
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        """
        # Filter out None values (patients without labels) and gently fix malformed items
        cleaned_batch = []
        for item in batch:
            if item is None:
                continue
            
            if not isinstance(item, dict):
                if not self._warned_once:
                    warnings.warn(
                        f"ClassificationCollator received a non-dict item. It will be skipped. "
                        f"Type: {type(item)}"
                    )
                    self._warned_once = True
                continue
            
            # If we have a label but no text, fall back to an empty string (rare edge case)
            if 'label' in item and 'text' not in item:
                if not self._warned_once:
                    warnings.warn(
                        "ClassificationCollator received an item with 'label' but no 'text'. "
                        "Using empty string as text for this item."
                    )
                    self._warned_once = True
                item = {**item, 'text': ""}
            
            # Require both keys after any fixes
            if 'text' not in item or 'label' not in item:
                if not self._warned_once:
                    warnings.warn(
                        f"ClassificationCollator received an item without required keys "
                        f"'text' and 'label'. It will be skipped. Example keys: {list(item.keys())}"
                    )
                    self._warned_once = True
                continue
            
            cleaned_batch.append(item)
        
        batch = cleaned_batch
        if not batch:
            return None
        
        # Extract text and labels
        texts = [item['text'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Clean up text - remove custom dataset format tokens and tokenizer special tokens
        # Only remove <end> - tokenizer will add proper special tokens
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        
        cleaned_texts = []
        for text in texts:
            # Remove custom tokens and any tokenizer EOS that might have been added by preprocessing
            text = text.replace('<end>', '')    
            if eos_token:
                text = text.replace(eos_token, '')
            cleaned_texts.append(text.strip())
        
        texts = cleaned_texts
        
        # Convert to binary labels if needed
        if self.binary_classification:
            labels = (labels > 0).long()
        
        # Tokenize each text individually WITHOUT padding, and handle EOS properly
        tokenizer_kwargs = {
            'padding': False,
            'truncation': False,  # We'll handle truncation manually
            'return_tensors': 'pt',
            'return_attention_mask': True,
            'add_special_tokens': True  # Adds BOS for decoder models
        }
        
        encoded_list = []
        for text in texts:
            # Tokenize the text
            encoded = self.tokenizer(text, **tokenizer_kwargs)
            
            # Add EOS token BEFORE any truncation
            if self.tokenizer.eos_token_id is not None:
                eos_token_id = self.tokenizer.eos_token_id
                eos_tensor = torch.tensor([[eos_token_id]], dtype=encoded['input_ids'].dtype)
                eos_mask = torch.ones((1, 1), dtype=encoded['attention_mask'].dtype)
                
                encoded['input_ids'] = torch.cat([encoded['input_ids'], eos_tensor], dim=1)
                encoded['attention_mask'] = torch.cat([encoded['attention_mask'], eos_mask], dim=1)
            
            # NOW truncate if sequence is too long (EOS is already at the end and will be preserved)
            seq_len = encoded['input_ids'].size(1)
            if self.max_length is not None and seq_len > self.max_length:
                # Truncate from the START to keep the END (most recent events + EOS)
                encoded['input_ids'] = encoded['input_ids'][:, -self.max_length:]
                encoded['attention_mask'] = encoded['attention_mask'][:, -self.max_length:]
                
                self._long_sequence_count += 1
                if not self._warned_once:
                    warnings.warn(
                        f"Found sequence with length {seq_len} exceeding max_length {self.max_length}. "
                        f"Truncating from the start (keeping most recent events and EOS token). "
                        f"This warning will only show once."
                    )
                    self._warned_once = True
            
            self._total_sequences += 1
            encoded_list.append(encoded)
        
        # NOW pad all sequences to the same length
        max_length_in_batch = max(enc['input_ids'].size(1) for enc in encoded_list)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for encoded in encoded_list:
            seq_len = encoded['input_ids'].size(1)
            padding_length = max_length_in_batch - seq_len
            
            if padding_length > 0:
                # Pad on the right
                padding_ids = torch.full((1, padding_length), pad_token_id, dtype=encoded['input_ids'].dtype)
                padding_mask = torch.zeros((1, padding_length), dtype=encoded['attention_mask'].dtype)
                
                padded_ids = torch.cat([encoded['input_ids'], padding_ids], dim=1)
                padded_mask = torch.cat([encoded['attention_mask'], padding_mask], dim=1)
            else:
                padded_ids = encoded['input_ids']
                padded_mask = encoded['attention_mask']
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        # Stack into batch tensors
        input_ids = torch.cat(padded_input_ids, dim=0)
        attention_mask = torch.cat(padded_attention_masks, dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_stats(self):
        """Return statistics about long sequences encountered."""
        return {
            'total_sequences': self._total_sequences,
            'long_sequences': self._long_sequence_count,
            'percentage_long': (self._long_sequence_count / self._total_sequences * 100) 
                               if self._total_sequences > 0 else 0
        }


