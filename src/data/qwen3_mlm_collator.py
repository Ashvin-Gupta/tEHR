"""
Data collator for masked language modeling (MLM) with Qwen3-Embedding.

This collator expects **pre-tokenized, packed** `input_ids` sequences:
  - Each item in the dataset should be a dict with an `input_ids` list of token IDs.
  - Sequences can span multiple patients; separator/EOS tokens between patients
    should already have been inserted upstream (e.g. via `pack_and_chunk_texts`).

Within each batch, we:
  - Pad sequences to the longest length in the batch.
  - Apply BERT-style 15% masking:
        * 80% -> [MASK] (if available) or random token fallback
        * 10% -> random token
        * 10% -> unchanged
  - Create `labels` with -100 on unmasked positions (ignored in loss).
"""

from typing import List, Dict, Any

import torch


class Qwen3MLMCollator:
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self.mask_token_id = getattr(tokenizer, "mask_token_id", None)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        if self.mask_token_id is None:
            print(
                "⚠️  Qwen3 tokenizer does not define a [MASK] token. "
                "MLM collator will use random-token replacement only."
            )

    def _mask_tokens(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        This mirrors the HuggingFace `DataCollatorForLanguageModeling` logic.
        """
        labels = inputs.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=inputs.device)

        # Do not mask padding tokens
        special_mask = inputs.eq(self.pad_token_id)

        # Also avoid masking special tokens provided by tokenizer
        if hasattr(self.tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
                for val in inputs
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask,
                dtype=torch.bool,
                device=inputs.device,
            )
            special_mask = special_mask | special_tokens_mask

        probability_matrix.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # If we have a [MASK] token, follow standard 80/10/10 rule
        if self.mask_token_id is not None:
            # 80% of the time, replace masked input tokens with [MASK]
            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool()
                & masked_indices
            )
            inputs[indices_replaced] = self.mask_token_id

            # 10% of the time, replace masked input tokens with random token
            indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool()
                & masked_indices
                & ~indices_replaced
            )
            random_words = torch.randint(
                low=0,
                high=len(self.tokenizer),
                size=labels.shape,
                dtype=torch.long,
                device=inputs.device,
            )
            inputs[indices_random] = random_words[indices_random]
            # The rest 10% we keep original token (inputs unchanged)
        else:
            # No explicit [MASK] token: replace all masked positions with random tokens
            random_words = torch.randint(
                low=0,
                high=len(self.tokenizer),
                size=labels.shape,
                dtype=torch.long,
                device=inputs.device,
            )
            inputs[masked_indices] = random_words[masked_indices]

        return inputs, labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Expect pre-tokenized input_ids lists (no padding yet)
        input_id_tensors = []
        for item in batch:
            ids = item["input_ids"]
            if isinstance(ids, torch.Tensor):
                input_id_tensors.append(ids.long())
            else:
                input_id_tensors.append(torch.tensor(ids, dtype=torch.long))

        # Dynamic right-padding within batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_id_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = (input_ids != self.pad_token_id).long()

        # Apply MLM masking
        input_ids, labels = self._mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

