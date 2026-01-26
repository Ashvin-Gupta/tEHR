"""
Data collator for LLM pretraining with optional temporal embeddings.

Similar to classification collator but designed for SFTTrainer pretraining tasks.
Handles text sequences without classification labels.
"""

import torch
from typing import List, Dict, Any, Optional
from src.data.temporal_utils import align_single_sequence


class PretrainingCollator:
    """
    Collate function for LLM pretraining tasks with optional temporal embeddings.
    
    Takes raw text from UnifiedEHRDataset and tokenizes it for batch processing.
    Optionally aligns and returns delta times for temporal embeddings.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length for truncation
        use_temporal_embeddings: If True, process and return delta times
        temporal_config: Config dict with 'time_scale' for log scaling
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        use_temporal_embeddings: bool = False,
        temporal_config: Optional[Dict] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_temporal_embeddings = use_temporal_embeddings
        self.temporal_config = temporal_config or {}
        self.time_scale = self.temporal_config.get('time_scale', 86400.0)  # Default: days
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples for pretraining.
        
        Args:
            batch: List of dicts with 'text' and optionally 'delta_times'
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels', and optionally 'delta_times'
        """
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # Extract text and optionally delta_times
        texts = [item['text'] for item in batch]
        delta_times_list = [item.get('delta_times') for item in batch] if self.use_temporal_embeddings else None
        
        # Clean up text - remove dataset format tokens
        texts = [text.replace('<start>', '').replace('<end>', '').strip() for text in texts]
        
        # Tokenize the text
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # For causal LM, labels are the same as input_ids (shifted internally by the model)
        labels = input_ids.clone()
        
        # Align delta times with tokenized sequences if using temporal embeddings
        delta_times_tensor = None
        if self.use_temporal_embeddings and delta_times_list is not None:
            aligned_delta_times = []
            max_length_in_batch = input_ids.shape[1]
            
            for text, delta_times in zip(texts, delta_times_list):
                if delta_times is None or len(delta_times) == 0:
                    # No delta times available - use zeros
                    aligned_deltas = [0.0] * max_length_in_batch
                else:
                    # Align delta times with tokenized sequence
                    aligned_deltas = align_single_sequence(text, delta_times, self.tokenizer)
                    
                    # Truncate/pad to match max_length_in_batch
                    if len(aligned_deltas) > max_length_in_batch:
                        aligned_deltas = aligned_deltas[:max_length_in_batch]
                    elif len(aligned_deltas) < max_length_in_batch:
                        aligned_deltas.extend([0.0] * (max_length_in_batch - len(aligned_deltas)))
                
                aligned_delta_times.append(aligned_deltas)
            
            # Convert to tensor and apply log scaling
            delta_times_tensor = torch.tensor(aligned_delta_times, dtype=torch.float32)
            # Apply log scaling: log(1 + delta_time / time_scale)
            delta_times_tensor = torch.log1p(delta_times_tensor / self.time_scale)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if delta_times_tensor is not None:
            result['delta_times'] = delta_times_tensor
        
        return result
