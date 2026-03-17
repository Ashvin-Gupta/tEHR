"""
Tests for src/training/training_utils.py

Covers:
- pack_and_chunk_texts:
    - All chunks are exactly chunk_size tokens long
    - Partial final chunk is dropped (no short chunks at end)
    - EOS token is appended between patient sequences
    - Returns a list of lists of ints
    - Empty input → empty output
- extract_text:
    - Returns a list of strings
    - Each returned string ends with the EOS token string
    - <end> tokens are stripped from the text
    - <HEADER_SPLIT> markers are stripped from the text
    - Leading '; ' is stripped
    - None items from the dataset are skipped
- compute_and_sort_by_length:
    - Output has same length as input
    - Sorted order is non-decreasing in token count
    - Bucketed shuffle doesn't lose any indices
- seed_all:
    - Calling with the same seed produces identical random outputs
"""
import pytest
import torch
import random
import numpy as np
from unittest.mock import MagicMock

from src.training.training_utils import (
    pack_and_chunk_texts,
    extract_text,
    compute_and_sort_by_length,
    seed_all,
)


# ---------------------------------------------------------------------------
# pack_and_chunk_texts
# ---------------------------------------------------------------------------

class TestPackAndChunkTexts:

    def test_all_chunks_are_exact_size(self, mock_tokenizer):
        texts = ["hello world"] * 20
        chunk_size = 10
        chunks = pack_and_chunk_texts(texts, mock_tokenizer, chunk_size)
        for chunk in chunks:
            assert len(chunk) == chunk_size, (
                f"Chunk length {len(chunk)} != chunk_size {chunk_size}"
            )

    def test_no_partial_final_chunk(self, mock_tokenizer):
        """The partial final chunk (< chunk_size) must be dropped."""
        texts = ["abc"] * 3  # each encodes to ~3 tokens; unlikely to divide evenly
        chunk_size = 8
        chunks = pack_and_chunk_texts(texts, mock_tokenizer, chunk_size)
        for chunk in chunks:
            assert len(chunk) == chunk_size

    def test_returns_list_of_lists(self, mock_tokenizer):
        texts = ["some text"] * 10
        chunks = pack_and_chunk_texts(texts, mock_tokenizer, 16)
        assert isinstance(chunks, list)
        assert all(isinstance(c, list) for c in chunks)
        assert all(isinstance(t, int) for c in chunks for t in c)

    def test_eos_appended_between_sequences(self, mock_tokenizer):
        """EOS (token_id=2) must appear between individual patient sequences."""
        texts = ["hello", "world", "foo"]
        chunk_size = 100  # large enough to keep all tokens in one chunk
        chunks = pack_and_chunk_texts(texts, mock_tokenizer, chunk_size)
        if not chunks:
            pytest.skip("Not enough tokens to fill a single chunk")
        flat = [t for chunk in chunks for t in chunk]
        assert mock_tokenizer.eos_token_id in flat, (
            "EOS token not found in packed output. "
            "Sequences must be separated by EOS."
        )

    def test_empty_input_returns_empty(self, mock_tokenizer):
        chunks = pack_and_chunk_texts([], mock_tokenizer, 16)
        assert chunks == []

    def test_token_count_consistent(self, mock_tokenizer):
        """Total tokens in output ≤ total tokens in input (some dropped for partial chunk)."""
        texts = ["abcde"] * 10
        chunk_size = 12
        chunks = pack_and_chunk_texts(texts, mock_tokenizer, chunk_size)
        total_out = sum(len(c) for c in chunks)
        assert total_out % chunk_size == 0


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:

    def _make_dataset(self, items):
        """Wrap a list into a mock dataset object."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(items))
        ds.__getitem__ = MagicMock(side_effect=lambda i: items[i])
        return ds

    def test_returns_list_of_strings(self, mock_tokenizer):
        ds = self._make_dataset([
            {"text": "hello world"},
            {"text": "another sentence"},
        ])
        result = extract_text(ds, mock_tokenizer)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_none_items_skipped(self, mock_tokenizer):
        ds = self._make_dataset([
            {"text": "valid text"},
            None,
            {"text": "also valid"},
        ])
        result = extract_text(ds, mock_tokenizer)
        assert len(result) == 2

    def test_eos_appended_to_each_sequence(self, mock_tokenizer):
        """Every returned string must end with the EOS token string.</s>"""
        ds = self._make_dataset([
            {"text": "patient one"},
            {"text": "patient two"},
        ])
        result = extract_text(ds, mock_tokenizer)
        for text in result:
            assert text.endswith(mock_tokenizer.eos_token), (
                f"Expected EOS '{mock_tokenizer.eos_token}' at end of: {repr(text)}"
            )

    def test_eos_not_duplicated(self, mock_tokenizer):
        """If the text already ends with EOS, it should not be appended again."""
        eos = mock_tokenizer.eos_token
        ds = self._make_dataset([{"text": f"patient text{eos}"}])
        result = extract_text(ds, mock_tokenizer)
        assert result[0].endswith(eos)
        assert result[0].count(eos) == 1

    def test_end_token_stripped(self, mock_tokenizer):
        ds = self._make_dataset([{"text": "hello <end> world"}])
        result = extract_text(ds, mock_tokenizer)
        assert "<end>" not in result[0]

    def test_header_split_stripped(self, mock_tokenizer):
        ds = self._make_dataset([{"text": "header <HEADER_SPLIT> body"}])
        result = extract_text(ds, mock_tokenizer)
        # In pretraining we remove the split marker entirely
        assert "<HEADER_SPLIT>" not in result[0]

    def test_leading_semicolon_stripped(self, mock_tokenizer):
        ds = self._make_dataset([{"text": "; actual content"}])
        result = extract_text(ds, mock_tokenizer)
        assert not result[0].startswith("; ")

    def test_empty_dataset(self, mock_tokenizer):
        ds = self._make_dataset([])
        result = extract_text(ds, mock_tokenizer)
        assert result == []


# ---------------------------------------------------------------------------
# compute_and_sort_by_length
# ---------------------------------------------------------------------------

class TestComputeAndSortByLength:

    def _make_dataset(self, texts):
        class FakeDataset:
            def __len__(self):
                return len(texts)
            def __getitem__(self, idx):
                return {"text": texts[idx]}
        return FakeDataset()

    def test_output_has_same_length_as_input(self, mock_tokenizer):
        ds = self._make_dataset(["short", "a medium length text", "tiny"])
        sorted_ds = compute_and_sort_by_length(ds, mock_tokenizer, shuffle_buckets=False)
        assert len(sorted_ds) == 3

    def test_sorted_order_is_non_decreasing(self, mock_tokenizer):
        texts = ["a long text that has many more tokens than the others",
                 "mid",
                 "x"]
        ds = self._make_dataset(texts)
        sorted_ds = compute_and_sort_by_length(ds, mock_tokenizer, shuffle_buckets=False)
        lengths = [
            len(mock_tokenizer.encode(sorted_ds[i]["text"], add_special_tokens=True))
            for i in range(len(sorted_ds))
        ]
        assert lengths == sorted(lengths), (
            f"Lengths not in ascending order: {lengths}"
        )

    def test_no_indices_lost(self, mock_tokenizer):
        texts = [f"text {i}" for i in range(20)]
        ds = self._make_dataset(texts)
        sorted_ds = compute_and_sort_by_length(ds, mock_tokenizer, shuffle_buckets=True)
        assert len(sorted_ds) == len(texts)


# ---------------------------------------------------------------------------
# seed_all
# ---------------------------------------------------------------------------

class TestSeedAll:

    def test_same_seed_produces_same_random(self):
        seed_all(42)
        a = random.random()
        seed_all(42)
        b = random.random()
        assert a == b

    def test_same_seed_produces_same_numpy_random(self):
        seed_all(123)
        a = np.random.rand()
        seed_all(123)
        b = np.random.rand()
        assert a == b

    def test_same_seed_produces_same_torch_random(self):
        seed_all(7)
        a = torch.rand(1).item()
        seed_all(7)
        b = torch.rand(1).item()
        assert a == b

    def test_different_seeds_produce_different_randoms(self):
        seed_all(1)
        a = random.random()
        seed_all(2)
        b = random.random()
        assert a != b
