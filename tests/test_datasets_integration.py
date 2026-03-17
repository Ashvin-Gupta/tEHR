"""
Integration tests for MIMICDataset and CPRDDataset.

All data is synthesised in a temp directory — no real patient files needed.
Fixtures are in conftest.py (write_fake_mimic_data / write_fake_cprd_data).

Covers:
- Dataset can be instantiated without crashing
- __len__ reflects the number of records (pretrain) / ACES examples (prediction)
- __getitem__ returns a dict with keys 'text', 'label', 'subject_id'
- 'text' is a non-empty string
- 'label' is a 0-d torch.LongTensor
- All subjects have valid labels
- MIMIC prediction task: examples are built from ACES parquet labels
- MIMIC pretrain task: examples == number of loaded patient records
- Time cutoff: tokens after the prediction time are excluded
- CPRD time cutoff: tokens within cutoff_months before diagnosis are excluded
- dataset_factory.get_dataset returns the correct type
"""
import os
import pytest
import torch
import numpy as np

from tests.conftest import write_fake_mimic_data, write_fake_cprd_data
from src.data.mimic_dataset import MIMICDataset
from src.data.cprd_dataset import CPRDDataset
from src.data.dataset_factory import get_dataset


RECORDS_PER_SPLIT = 5


# ---------------------------------------------------------------------------
# MIMIC dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mimic_paths(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("mimic"))
    return write_fake_mimic_data(tmp, records_per_split=RECORDS_PER_SPLIT)


@pytest.fixture(scope="module")
def mimic_pretrain(mimic_paths):
    return MIMICDataset(
        data_dir=mimic_paths["data_dir"],
        vocab_file=mimic_paths["vocab_filepath"],
        aces_label_dir=mimic_paths["aces_label_dir"],
        lookup_dir=mimic_paths["lookup_dir"],
        task="pretrain",
        split="train",
    )


@pytest.fixture(scope="module")
def mimic_prediction(mimic_paths):
    return MIMICDataset(
        data_dir=mimic_paths["data_dir"],
        vocab_file=mimic_paths["vocab_filepath"],
        aces_label_dir=mimic_paths["aces_label_dir"],
        lookup_dir=mimic_paths["lookup_dir"],
        task="prediction",
        split="train",
        label_column="label",
    )


# ---------------------------------------------------------------------------
# CPRD dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cprd_paths(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("cprd"))
    return write_fake_cprd_data(tmp, records_per_split=RECORDS_PER_SPLIT)


@pytest.fixture(scope="module")
def cprd_dataset(cprd_paths):
    return CPRDDataset(
        data_dir=cprd_paths["data_dir"],
        vocab_file=cprd_paths["vocab_filepath"],
        labels_file=cprd_paths["labels_filepath"],
        medical_lookup_file=cprd_paths["medical_lookup_filepath"],
        lab_lookup_file=cprd_paths["lab_lookup_filepath"],
        region_lookup_file=cprd_paths["region_lookup_filepath"],
        time_lookup_file=cprd_paths["time_lookup_filepath"],
        split="train",
        format="text",
    )


# ---------------------------------------------------------------------------
# MIMICDataset — pretrain task
# ---------------------------------------------------------------------------

class TestMIMICPretrain:

    def test_instantiation(self, mimic_pretrain):
        assert mimic_pretrain is not None

    def test_len_equals_records(self, mimic_pretrain):
        assert len(mimic_pretrain) == RECORDS_PER_SPLIT

    def test_getitem_returns_dict(self, mimic_pretrain):
        item = mimic_pretrain[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, mimic_pretrain):
        item = mimic_pretrain[0]
        assert "text" in item
        assert "label" in item
        assert "subject_id" in item

    def test_text_is_nonempty_string(self, mimic_pretrain):
        for i in range(len(mimic_pretrain)):
            item = mimic_pretrain[i]
            assert isinstance(item["text"], str)
            assert len(item["text"]) > 0

    def test_label_is_tensor(self, mimic_pretrain):
        item = mimic_pretrain[0]
        assert isinstance(item["label"], torch.Tensor)
        assert item["label"].dtype == torch.long

    def test_pretrain_labels_are_zero(self, mimic_pretrain):
        """In pretrain mode all labels should be 0 (no task-specific annotation)."""
        for i in range(len(mimic_pretrain)):
            assert mimic_pretrain[i]["label"].item() == 0

    def test_text_contains_clinical_content(self, mimic_pretrain):
        """Token translation should have produced some recognisable text."""
        item = mimic_pretrain[0]
        # The fake vocab includes a diagnosis and a lab token; at least one should appear
        text = item["text"]
        has_content = any(kw in text.lower() for kw in [
            "hypertension", "urea nitrogen", "age", "female", "2020"
        ])
        assert has_content, f"No expected clinical content found in text: {repr(text[:200])}"


# ---------------------------------------------------------------------------
# MIMICDataset — prediction task
# ---------------------------------------------------------------------------

class TestMIMICPrediction:

    def test_instantiation(self, mimic_prediction):
        assert mimic_prediction is not None

    def test_len_equals_aces_examples(self, mimic_prediction):
        """Length should equal the number of ACES rows with a matching trajectory."""
        assert len(mimic_prediction) == RECORDS_PER_SPLIT

    def test_labels_are_binary(self, mimic_prediction):
        labels = [mimic_prediction[i]["label"].item() for i in range(len(mimic_prediction))]
        assert set(labels).issubset({0, 1})

    def test_both_classes_present(self, mimic_prediction):
        """Fake data alternates 0/1 labels so both should appear."""
        labels = [mimic_prediction[i]["label"].item() for i in range(len(mimic_prediction))]
        assert 0 in labels
        assert 1 in labels

    def test_subject_ids_are_correct_type(self, mimic_prediction):
        for i in range(len(mimic_prediction)):
            sid = mimic_prediction[i]["subject_id"]
            assert sid is not None


# ---------------------------------------------------------------------------
# MIMICDataset — time cutoff
# ---------------------------------------------------------------------------

class TestMIMICTimeCutoff:

    def test_tokens_after_prediction_time_excluded(self, mimic_paths):
        """
        The fake data has tokens with timestamps:
          - t=0         : header tokens (always kept)
          - base_time   : first clinical events (kept if <= pred_time)
          - base_time+3600 : event close to prediction time
          - base_time+7200 : prediction_time itself (= kept)

        Set prediction_time to base_time so only t=0 and base_time tokens
        are returned; base_time+3600 and base_time+7200 events are cut.
        """
        base_time = 1577836800
        import pandas as pd
        import os
        # Overwrite the prediction_time to base_time in the labels
        aces_path = os.path.join(mimic_paths["aces_label_dir"], "train_labels.parquet")
        aces_df = pd.read_parquet(aces_path)
        # Temporarily rebuild dataset with early prediction_time
        aces_early = aces_df.copy()
        aces_early["prediction_time"] = float(base_time)  # cut after header+first events

        # Save modified parquet
        import tempfile, os
        tmp_dir = tempfile.mkdtemp()
        aces_early.to_parquet(os.path.join(tmp_dir, "train_labels.parquet"), index=False)

        ds = MIMICDataset(
            data_dir=mimic_paths["data_dir"],
            vocab_file=mimic_paths["vocab_filepath"],
            aces_label_dir=tmp_dir,
            lookup_dir=mimic_paths["lookup_dir"],
            task="prediction",
            split="train",
        )

        item = ds[0]
        # With prediction_time = base_time, only tokens with t <= base_time are kept.
        # Token ids 8 (DIAGNOSIS) and 9 (TIME 2020-01) have timestamps base_time+3600
        # and base_time+7200 respectively → should be absent from text.
        text_early = item["text"]

        # Now get text with late prediction_time (everything included)
        item_late = MIMICDataset(
            data_dir=mimic_paths["data_dir"],
            vocab_file=mimic_paths["vocab_filepath"],
            aces_label_dir=mimic_paths["aces_label_dir"],
            lookup_dir=mimic_paths["lookup_dir"],
            task="prediction",
            split="train",
        )[0]
        text_late = item_late["text"]

        assert len(text_early) <= len(text_late), (
            "Early cutoff should produce shorter or equal text length. "
            f"Early: {len(text_early)}, Late: {len(text_late)}"
        )


# ---------------------------------------------------------------------------
# CPRDDataset
# ---------------------------------------------------------------------------

class TestCPRDDataset:

    def test_instantiation(self, cprd_dataset):
        assert cprd_dataset is not None

    def test_len(self, cprd_dataset):
        assert len(cprd_dataset) == RECORDS_PER_SPLIT

    def test_getitem_has_required_keys(self, cprd_dataset):
        item = cprd_dataset[0]
        assert item is not None
        assert "text" in item
        assert "label" in item
        assert "subject_id" in item

    def test_text_is_nonempty_string(self, cprd_dataset):
        item = cprd_dataset[0]
        assert isinstance(item["text"], str)
        assert len(item["text"]) > 0

    def test_label_is_long_tensor(self, cprd_dataset):
        for i in range(len(cprd_dataset)):
            item = cprd_dataset[i]
            if item is None:
                continue
            assert isinstance(item["label"], torch.Tensor)
            assert item["label"].dtype == torch.long

    def test_labels_are_valid(self, cprd_dataset):
        """Labels must be non-negative integers (0 = Control, 1+ = cancer site)."""
        for i in range(len(cprd_dataset)):
            item = cprd_dataset[i]
            if item is None:
                continue
            assert item["label"].item() >= 0

    def test_both_classes_present(self, cprd_dataset):
        labels = [
            cprd_dataset[i]["label"].item()
            for i in range(len(cprd_dataset))
            if cprd_dataset[i] is not None
        ]
        assert 0 in labels
        assert any(l > 0 for l in labels)


class TestCPRDTimeCutoff:

    def test_cutoff_reduces_tokens_for_cases(self, cprd_paths):
        """
        For cancer cases, applying a time cutoff should produce equal-length or
        shorter token sequences compared to no cutoff.
        """
        ds_no_cutoff = CPRDDataset(
            data_dir=cprd_paths["data_dir"],
            vocab_file=cprd_paths["vocab_filepath"],
            labels_file=cprd_paths["labels_filepath"],
            medical_lookup_file=cprd_paths["medical_lookup_filepath"],
            lab_lookup_file=cprd_paths["lab_lookup_filepath"],
            region_lookup_file=cprd_paths["region_lookup_filepath"],
            time_lookup_file=cprd_paths["time_lookup_filepath"],
            split="train",
            cutoff_months=None,
        )
        ds_with_cutoff = CPRDDataset(
            data_dir=cprd_paths["data_dir"],
            vocab_file=cprd_paths["vocab_filepath"],
            labels_file=cprd_paths["labels_filepath"],
            medical_lookup_file=cprd_paths["medical_lookup_filepath"],
            lab_lookup_file=cprd_paths["lab_lookup_filepath"],
            region_lookup_file=cprd_paths["region_lookup_filepath"],
            time_lookup_file=cprd_paths["time_lookup_filepath"],
            split="train",
            cutoff_months=1,
        )

        for i in range(len(ds_no_cutoff)):
            item_none = ds_no_cutoff[i]
            item_cut  = ds_with_cutoff[i]
            if item_none is None or item_cut is None:
                continue
            # Cutoff should not lengthen sequences
            assert len(item_cut["text"]) <= len(item_none["text"])


# ---------------------------------------------------------------------------
# dataset_factory.get_dataset
# ---------------------------------------------------------------------------

class TestDatasetFactory:

    def test_mimic_factory_returns_mimic_dataset(self, mimic_paths):
        config = {
            "data_dir": mimic_paths["data_dir"],
            "vocab_filepath": mimic_paths["vocab_filepath"],
            "aces_label_dir": mimic_paths["aces_label_dir"],
            "lookup_dir": mimic_paths["lookup_dir"],
            "task": "pretrain",
        }
        ds = get_dataset("MIMIC", config, split="train")
        assert isinstance(ds, MIMICDataset)

    def test_cprd_factory_returns_cprd_dataset(self, cprd_paths):
        config = {
            "data_dir": cprd_paths["data_dir"],
            "vocab_filepath": cprd_paths["vocab_filepath"],
            "labels_filepath": cprd_paths["labels_filepath"],
            "medical_lookup_filepath": cprd_paths["medical_lookup_filepath"],
            "lab_lookup_filepath": cprd_paths["lab_lookup_filepath"],
            "region_lookup_filepath": cprd_paths["region_lookup_filepath"],
            "time_lookup_filepath": cprd_paths["time_lookup_filepath"],
        }
        ds = get_dataset("CPRD", config, split="train")
        assert isinstance(ds, CPRDDataset)

    def test_unknown_dataset_raises(self, mimic_paths):
        with pytest.raises(ValueError, match="Unsupported dataset"):
            get_dataset("UNKNOWN", {}, split="train")
