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
        self._missing_text_count = 0
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        """
        # 1. Filter out None values and malformed items
        cleaned_batch = []
        for i, item in enumerate(batch):
            if item is None:
                continue
            
            if not isinstance(item, dict):
                # Only warn once
                if not self._warned_once:
                    warnings.warn(f"Skipping non-dict item at index {i}. Type: {type(item)}")
                    self._warned_once = True
                continue
            
            # Check for required keys (Helpful debug info included)
            if 'text' not in item or 'label' not in item:
                self._missing_text_count += 1
                if not self._warned_once:
                    warnings.warn(
                        f"Skipping malformed item. Missing 'text' or 'label'. "
                        f"Found keys: {list(item.keys())}"
                    )
                    self._warned_once = True
                continue
            
            cleaned_batch.append(item)
        
        # 2. Handle Empty Batch (Prevents RuntimeError)
        if not cleaned_batch:
            return None
        
        # Extract text and labels
        texts = [item['text'] for item in cleaned_batch]
        labels = torch.stack([item['label'] for item in cleaned_batch])
        
        # Clean up text
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
        cleaned_texts = []
        for text in texts:
            text = text.replace('<end>', '')    
            if eos_token:
                text = text.replace(eos_token, '')
            cleaned_texts.append(text.strip())
        texts = cleaned_texts
        
        # Convert to binary labels if needed
        if self.binary_classification:
            labels = (labels > 0).long()
        
        # Tokenize WITHOUT padding initially
        tokenizer_kwargs = {
            'padding': False,
            'truncation': False,
            'return_tensors': 'pt',
            'return_attention_mask': True,
            'add_special_tokens': True 
        }
        
        encoded_list = []
        for text in texts:
            encoded = self.tokenizer(text, **tokenizer_kwargs)
            
            # Manually add EOS if not present (Important for Llama, usually auto for BERT)
            if self.tokenizer.eos_token_id is not None:
                last_token = encoded['input_ids'][0, -1]
                if last_token != self.tokenizer.eos_token_id:
                    eos_tensor = torch.tensor([[self.tokenizer.eos_token_id]], dtype=encoded['input_ids'].dtype)
                    eos_mask = torch.ones((1, 1), dtype=encoded['attention_mask'].dtype)
                    encoded['input_ids'] = torch.cat([encoded['input_ids'], eos_tensor], dim=1)
                    encoded['attention_mask'] = torch.cat([encoded['attention_mask'], eos_mask], dim=1)
            
            # --- UPDATED TRUNCATION LOGIC (Fixes ClinicalBERT) ---
            seq_len = encoded['input_ids'].size(1)
            if self.max_length is not None and seq_len > self.max_length:
                # Detect [CLS] token at the start (index 0)
                has_cls = (self.tokenizer.cls_token_id is not None) and \
                          (encoded['input_ids'][0, 0] == self.tokenizer.cls_token_id)
                
                if has_cls:
                    # ClinicalBERT: Keep [CLS] + [Most Recent Events]
                    trunc_len = self.max_length - 1
                    
                    cls_id = encoded['input_ids'][:, :1]      # The [CLS]
                    cls_mask = encoded['attention_mask'][:, :1]
                    
                    recent_ids = encoded['input_ids'][:, -trunc_len:]
                    recent_mask = encoded['attention_mask'][:, -trunc_len:]
                    
                    encoded['input_ids'] = torch.cat([cls_id, recent_ids], dim=1)
                    encoded['attention_mask'] = torch.cat([cls_mask, recent_mask], dim=1)
                else:
                    # Standard LLM: Just keep the end
                    encoded['input_ids'] = encoded['input_ids'][:, -self.max_length:]
                    encoded['attention_mask'] = encoded['attention_mask'][:, -self.max_length:]
                
                self._long_sequence_count += 1
            # -----------------------------------------------------
            
            self._total_sequences += 1
            encoded_list.append(encoded)
        
        # Dynamic Padding
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
        
        return {
            'input_ids': torch.cat(padded_input_ids, dim=0),
            'attention_mask': torch.cat(padded_attention_masks, dim=0),
            'labels': labels
        }
    
    def get_stats(self):
        """Return statistics about long sequences and malformed samples encountered."""
        return {
            'total_sequences': self._total_sequences,
            'long_sequences': self._long_sequence_count,
            'percentage_long': (self._long_sequence_count / self._total_sequences * 100) 
                               if self._total_sequences > 0 else 0,
            'missing_text_samples': self._missing_text_count,
        }


