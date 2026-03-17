"""
Tests for ClassificationCollator.

Covers:
- Output keys: input_ids, attention_mask, labels always present
- Batch dimension matches number of valid input items
- Labels are binarised (any label > 0 → 1)
- Padding: all sequences in a batch are padded to the same length
- Attention mask: 1 for real tokens, 0 for padding
- Truncation: output never exceeds max_length
- Last real token is EOS (both HEADER_SPLIT and fallback paths)
- HEADER_SPLIT smart truncation: header is always preserved
- Items without a 'label' key are silently skipped
- None items are silently skipped
- Empty batch → dummy batch returned (DDP failsafe)
- get_stats() accumulates totals correctly
"""
import pytest
import torch

from src.data.classification_collator import ClassificationCollator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item(text: str, label: int) -> dict:
    return {"text": text, "label": torch.tensor(label, dtype=torch.long)}


def last_real_token(input_ids_row: torch.Tensor, pad_id: int) -> int:
    """Return the last non-padding token id in a 1-D tensor."""
    mask = input_ids_row != pad_id
    if not mask.any():
        return pad_id
    return input_ids_row[mask.nonzero(as_tuple=True)[0][-1]].item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassificationCollator:

    @pytest.fixture
    def collator(self, mock_tokenizer):
        return ClassificationCollator(mock_tokenizer, max_length=64)

    # --- basic output shape ---

    def test_output_keys(self, collator):
        batch = [make_item("hello world", 0)]
        out = collator(batch)
        assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_batch_size_single(self, collator):
        out = collator([make_item("hello", 1)])
        assert out["input_ids"].shape[0] == 1
        assert out["labels"].shape[0] == 1

    def test_batch_size_multiple(self, collator):
        batch = [make_item("hello", 0), make_item("world", 1), make_item("foo", 0)]
        out = collator(batch)
        assert out["input_ids"].shape[0] == 3
        assert out["labels"].shape[0] == 3

    # --- label binarisation ---

    def test_label_zero_stays_zero(self, collator):
        out = collator([make_item("text", 0)])
        assert out["labels"][0].item() == 0

    def test_label_one_stays_one(self, collator):
        out = collator([make_item("text", 1)])
        assert out["labels"][0].item() == 1

    def test_label_multi_class_binarised(self, collator):
        """Any label > 0 must be mapped to 1 (binary classification)."""
        batch = [make_item("a", 2), make_item("b", 5), make_item("c", 0)]
        out = collator(batch)
        assert out["labels"][0].item() == 1
        assert out["labels"][1].item() == 1
        assert out["labels"][2].item() == 0

    # --- truncation ---

    def test_output_never_exceeds_max_length(self, mock_tokenizer):
        collator = ClassificationCollator(mock_tokenizer, max_length=16)
        long_text = "a" * 200
        out = collator([make_item(long_text, 0)])
        assert out["input_ids"].shape[1] <= 16

    # --- EOS token at end of real sequence ---

    def test_last_real_token_is_eos_fallback_path(self, mock_tokenizer):
        """When no HEADER_SPLIT present, last real token must be EOS."""
        collator = ClassificationCollator(mock_tokenizer, max_length=64)
        out = collator([make_item("some clinical text", 0)])
        row = out["input_ids"][0]
        assert last_real_token(row, mock_tokenizer.pad_token_id) == mock_tokenizer.eos_token_id

    def test_last_real_token_is_eos_header_split_path(self, mock_tokenizer):
        """When HEADER_SPLIT is present, last real token must still be EOS."""
        collator = ClassificationCollator(mock_tokenizer, max_length=64)
        # Use the format the collator expects: space on both sides
        text = "Age; Female;  <HEADER_SPLIT>  Hypertension; 2020-01; "
        out = collator([make_item(text, 0)])
        row = out["input_ids"][0]
        assert last_real_token(row, mock_tokenizer.pad_token_id) == mock_tokenizer.eos_token_id

    # --- padding ---

    def test_all_rows_same_length_after_padding(self, collator):
        batch = [
            make_item("short", 0),
            make_item("a much longer piece of text for testing", 1),
            make_item("medium length text", 0),
        ]
        out = collator(batch)
        lengths = set(out["input_ids"][i].shape[0] for i in range(len(batch)))
        assert len(lengths) == 1, f"Expected all rows same length, got lengths: {lengths}"

    def test_padding_fills_with_pad_id(self, mock_tokenizer):
        collator = ClassificationCollator(mock_tokenizer, max_length=64)
        batch = [make_item("hi", 0), make_item("a much longer text sample", 1)]
        out = collator(batch)
        short_row = out["input_ids"][0]
        # Shorter row should have trailing pad tokens
        assert (short_row == mock_tokenizer.pad_token_id).any()

    # --- attention mask ---

    def test_attention_mask_shape_matches_input_ids(self, collator):
        batch = [make_item("hello", 0), make_item("world again", 1)]
        out = collator(batch)
        assert out["attention_mask"].shape == out["input_ids"].shape

    def test_attention_mask_ones_for_real_tokens(self, mock_tokenizer):
        collator = ClassificationCollator(mock_tokenizer, max_length=64)
        batch = [make_item("short", 0), make_item("a longer piece of text here", 1)]
        out = collator(batch)
        for i in range(out["input_ids"].shape[0]):
            for j in range(out["input_ids"].shape[1]):
                token_id = out["input_ids"][i, j].item()
                mask_val = out["attention_mask"][i, j].item()
                if token_id == mock_tokenizer.pad_token_id:
                    assert mask_val == 0, f"Padding token at [{i},{j}] should have mask=0"
                else:
                    assert mask_val == 1, f"Real token at [{i},{j}] should have mask=1"

    # --- HEADER_SPLIT smart truncation ---

    def test_header_split_header_always_present(self, mock_tokenizer):
        """
        Header tokens must always appear even when the body is very long.
        The collator should keep header + tail-of-body, discarding middle body.
        """
        collator = ClassificationCollator(mock_tokenizer, max_length=20)
        header = "Age Female "
        # Make the body long enough that standard truncation would drop the header
        body = "clinical event " * 30
        text = f"{header} <HEADER_SPLIT> {body}"
        out = collator([make_item(text, 0)])
        # Decode back: the header chars should be present in the token ids
        decoded_ids = out["input_ids"][0].tolist()
        header_ids = mock_tokenizer.encode(header, add_special_tokens=False)
        # At least the first few header tokens should be present
        for hid in header_ids[:3]:
            assert hid in decoded_ids, (
                f"Header token {hid} missing from output after truncation. "
                "Smart truncation should preserve header at the cost of body middle."
            )

    # --- filtering bad items ---

    def test_items_without_label_skipped(self, collator):
        batch = [{"text": "some text"}, make_item("labelled", 1)]
        out = collator(batch)
        assert out["labels"].shape[0] == 1

    def test_none_items_skipped(self, collator):
        batch = [None, make_item("valid", 0), None]
        out = collator(batch)
        assert out["labels"].shape[0] == 1

    # --- empty batch failsafe ---

    def test_empty_batch_returns_dummy(self, collator):
        """All-None batch must return a dummy batch, not crash (DDP safeguard)."""
        out = collator([None, None])
        assert "input_ids" in out
        assert "labels" in out
        assert out["labels"].shape[0] == 1  # single dummy sample

    def test_empty_batch_dummy_label_is_zero(self, collator):
        out = collator([None])
        assert out["labels"][0].item() == 0

    # --- stats ---

    def test_get_stats_accumulates(self, collator):
        collator([make_item("text a", 0), make_item("text b", 1)])
        collator([make_item("text c", 0)])
        stats = collator.get_stats()
        assert stats["total"] == 3

    def test_get_stats_zero_initially(self, mock_tokenizer):
        fresh = ClassificationCollator(mock_tokenizer, max_length=64)
        stats = fresh.get_stats()
        assert stats["total"] == 0
        assert stats["long"] == 0
