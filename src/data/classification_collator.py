# src/data/classification_collator.py

import torch
from typing import List, Dict, Any
import warnings

class ClassificationCollator:
    """
    Robust Data Collator for LLM & BERT classification tasks.
    
    Features:
    - Handles 'text' or 'input_ids' input.
    - Robust truncation (preserves [CLS] for BERT, keeps most recent events).
    - DDP Failsafe (returns dummy batch instead of crashing on empty data).
    - Binary label conversion.
    """
    
    def __init__(self, tokenizer, max_length: int = 512, binary_classification: bool = True, 
                 truncation: bool = True, handle_long_sequences: str = 'warn'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.binary_classification = binary_classification
        self.truncation = truncation
        self.handle_long_sequences = handle_long_sequences
        
        # Statistics & State
        self._warned_once = False
        self._warned_missing_text = False
        self._long_sequence_count = 0
        self._total_sequences = 0
        # Debugging: how many sample trajectories to print
        self._debug_samples_printed = 0
        self._max_debug_samples = 5
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        """
        raise RuntimeError("DEBUG: ClassificationCollator.__call__ was invoked")
        # 1. Filter None items
        batch = [item for item in batch if item is not None]
        
        input_ids_list = []
        labels_list = []
        
        # Cache special tokens
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for item in batch:
          
            self._total_sequences += 1
            
            # --- Label Handling ---
            if 'label' not in item:
                print(f"Item without label: {item}")
                continue # Skip items without labels
                
            label = item['label']
            if self.binary_classification:
                # Handle both scalar tensors and python numbers
                if isinstance(label, torch.Tensor):
                    is_pos = (label.item() > 0)
                else:
                    is_pos = (label > 0)
                label = 1 if is_pos else 0
            
            # --- Input Handling ---
            ids = []
            if 'input_ids' in item:
                # Pre-tokenized
                val = item['input_ids']
                ids = val.tolist() if isinstance(val, torch.Tensor) else val
            else:
                # Raw text
                # Robust get: defaults to empty string if 'text' is missing
                text = item.get('text', '')
                
                if 'text' not in item and not self._warned_missing_text:
                    warnings.warn(
                        f"Item missing 'text' key. Found keys: {list(item.keys())}. "
                        "Using empty string. (This warning shows once)"
                    )
                    self._warned_missing_text = True
                
                # Cleanup
                text = text.replace('<start>', '').replace('<end>', '').strip()
                ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # --- EOS Enforcement ---
            # Ensure sequence ends with EOS so "recent events" logic works
            if eos_id is not None:
                if not ids or ids[-1] != eos_id:
                    ids.append(eos_id)
            
            # --- Truncation (The Critical Part) ---
            if self.max_length is not None and len(ids) > self.max_length:
                self._long_sequence_count += 1
                
                # Check for CLS at the start (ClinicalBERT/BERT)
                has_cls = (cls_id is not None) and (len(ids) > 0) and (ids[0] == cls_id)
                
                if has_cls:
                    # Keep [CLS] + [Most Recent Events (End)]
                    # We take the CLS, and then the last (max_len - 1) tokens
                    ids = [ids[0]] + ids[-(self.max_length - 1):]
                else:
                    # Standard LLM: Just keep the end
                    ids = ids[-self.max_length:]
            
            # --- DEBUG: Print a few sample trajectories seen by the model ---
            if self._debug_samples_printed < self._max_debug_samples:
                try:
                    decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
                    print("\n" + "=" * 80)
                    print("CLASSIFICATION COLLATOR DEBUG - SAMPLE TRAJECTORY")
                    print("=" * 80)
                    print(f"  Raw label: {item['label']}  →  Binary label: {label}")
                    print(f"  Sequence length (tokens): {len(ids)}")
                    print(f"  First 20 token IDs: {ids[:20]}")
                    print(f"  Last 20 token IDs:  {ids[-20:]}")
                    print("  Decoded text (truncated to 500 chars):")
                    print("  " + decoded[:500] + ("..." if len(decoded) > 500 else ""))
                    print("=" * 80 + "\n")
                except Exception as e:
                    print(f"CLASSIFICATION COLLATOR DEBUG: Failed to decode sample: {e}")
                self._debug_samples_printed += 1
            
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(label)
        
        # 2. DDP FAILSAFE: Handle Empty Batches
        # Prevents "TypeError: argument of type 'NoneType' is not iterable"
        if not input_ids_list:
            if not self._warned_once:
                warnings.warn("Empty batch produced! Returning dummy batch to prevent crash.")
                self._warned_once = True
                
            dummy_ids = torch.tensor([pad_id], dtype=torch.long)
            # Return a single dummy sample (label 0)
            return {
                'input_ids': dummy_ids.unsqueeze(0),
                'attention_mask': torch.ones((1, 1), dtype=torch.long),
                'labels': torch.tensor([0], dtype=torch.long)
            }
            
        # 3. Dynamic Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=pad_id
        )
        
        # Create mask (1 for real, 0 for pad)
        attention_mask = (input_ids != pad_id).long()
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }

    def get_stats(self):
        return {
            'total': self._total_sequences,
            'long': self._long_sequence_count,
            'pct_long': (self._long_sequence_count / self._total_sequences * 100) if self._total_sequences else 0
        }