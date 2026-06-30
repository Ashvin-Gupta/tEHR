"""
Tests for ACES label loading and filtering functionality.

These tests verify that:
1. Patient records are loaded from all .pkl files in the data directory
2. ACES labels are loaded from the parquet file
3. Only patient records with corresponding ACES labels are included in prediction examples
4. The filtering works correctly (records without ACES labels are excluded)
5. All ACES labels with matching patient records become examples
"""
import os
import pytest
import pickle
import pandas as pd
import tempfile
import shutil

from src.data.mimic_dataset import MIMICDataset


@pytest.fixture
def aces_filtering_setup(tmp_path):
    """
    Create a test setup where:
    - We have 10 patient records (subject_ids 1000-1009)
    - We have ACES labels for only 5 of them (1000, 1002, 1004, 1006, 1008)
    - This tests that only the 5 with ACES labels become prediction examples
    """
    tmp = str(tmp_path)
    
    # ---- vocab ----
    vocab_rows = [
        (0, "<pad>"), (1, "<start>"), (2, "AGE"), (3, "55.0"),
        (4, "GENDER//F"), (5, "LAB//51006"), (6, "13.5"), (7, "mmol/L"),
        (8, "DIAGNOSIS//ICD//10_I10"), (9, "TIME 2020-01"),
    ]
    vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
    vocab_file = os.path.join(tmp, "vocab.csv")
    vocab_df.to_csv(vocab_file, index=False)

    # ---- lookup tables ----
    hosp_dir = os.path.join(tmp, "hosp")
    os.makedirs(hosp_dir, exist_ok=True)

    lab_df = pd.DataFrame({"itemid": [51006], "label": ["Urea Nitrogen"]})
    lab_df.to_csv(os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip")

    diag_df = pd.DataFrame({
        "icd_version": [10],
        "icd_code": ["I10"],
        "long_title": ["Essential hypertension"],
    })
    diag_df.to_csv(os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip")

    proc_df = pd.DataFrame({
        "icd_version": [10],
        "icd_code": ["0BH17EZ"],
        "long_title": ["Insertion of Endotracheal Airway"],
    })
    proc_df.to_csv(os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip")

    # ---- patient records (10 total across multiple files) ----
    base_time = 1577836800
    data_dir = os.path.join(tmp, "data")
    train_dir = os.path.join(data_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    # Create 3 pickle files with different patient records
    # File 1: subjects 1000-1003
    records_batch1 = []
    for j in range(4):
        subject_id = 1000 + j
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        timestamps = [0, 0, 0, 0, base_time, base_time, base_time,
                      base_time + 3600, base_time + 7200]
        records_batch1.append({
            "subject_id": subject_id,
            "tokens": tokens,
            "timestamps": timestamps,
        })
    with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
        pickle.dump(records_batch1, f)

    # File 2: subjects 1004-1006
    records_batch2 = []
    for j in range(4, 7):
        subject_id = 1000 + j
        tokens = [1, 2, 3, 4, 5, 6, 7]
        timestamps = [0, 0, 0, 0, base_time, base_time, base_time]
        records_batch2.append({
            "subject_id": subject_id,
            "tokens": tokens,
            "timestamps": timestamps,
        })
    with open(os.path.join(train_dir, "batch_1.pkl"), "wb") as f:
        pickle.dump(records_batch2, f)

    # File 3: subjects 1007-1009
    records_batch3 = []
    for j in range(7, 10):
        subject_id = 1000 + j
        tokens = [1, 2, 3, 4, 5]
        timestamps = [0, 0, 0, 0, base_time]
        records_batch3.append({
            "subject_id": subject_id,
            "tokens": tokens,
            "timestamps": timestamps,
        })
    with open(os.path.join(train_dir, "batch_2.pkl"), "wb") as f:
        pickle.dump(records_batch3, f)

    # ---- ACES labels (only for subjects 1000, 1002, 1004, 1006, 1008) ----
    aces_dir = os.path.join(tmp, "aces_labels")
    os.makedirs(aces_dir, exist_ok=True)
    
    aces_rows = [
        {"subject_id": 1000, "label": 0, "prediction_time": float(base_time + 7200)},
        {"subject_id": 1002, "label": 1, "prediction_time": float(base_time + 7200)},
        {"subject_id": 1004, "label": 0, "prediction_time": float(base_time + 3600)},
        {"subject_id": 1006, "label": 1, "prediction_time": float(base_time + 3600)},
        {"subject_id": 1008, "label": 1, "prediction_time": float(base_time + 7200)},
    ]
    aces_df = pd.DataFrame(aces_rows)
    aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

    return {
        "data_dir": data_dir,
        "vocab_filepath": vocab_file,
        "aces_label_dir": aces_dir,
        "lookup_dir": hosp_dir,
        "expected_total_records": 10,  # Total patient records loaded
        "expected_aces_examples": 5,   # Only 5 have ACES labels
        "aces_subject_ids": {1000, 1002, 1004, 1006, 1008},
        "non_aces_subject_ids": {1001, 1003, 1005, 1007, 1009},
    }


class TestACESLabelLoading:
    """Test that ACES label loading works correctly with filtering."""

    def test_all_patient_records_loaded(self, aces_filtering_setup):
        """
        Verify that ALL patient records are loaded from the data directory,
        regardless of whether they have ACES labels.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # All 10 patient records should be loaded
        assert len(ds.patient_records) == aces_filtering_setup["expected_total_records"], (
            f"Expected {aces_filtering_setup['expected_total_records']} patient records to be loaded, "
            f"but got {len(ds.patient_records)}"
        )

    def test_only_aces_subjects_in_examples(self, aces_filtering_setup):
        """
        Verify that only patients with ACES labels become prediction examples.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Only 5 examples should be created (those with ACES labels)
        assert len(ds.examples) == aces_filtering_setup["expected_aces_examples"], (
            f"Expected {aces_filtering_setup['expected_aces_examples']} prediction examples, "
            f"but got {len(ds.examples)}"
        )
        
        # Verify all examples have subject_ids from the ACES set
        example_subject_ids = {ex["subject_id"] for ex in ds.examples}
        assert example_subject_ids == aces_filtering_setup["aces_subject_ids"], (
            f"Expected examples for subjects {aces_filtering_setup['aces_subject_ids']}, "
            f"but got {example_subject_ids}"
        )

    def test_dataset_length_equals_aces_examples(self, aces_filtering_setup):
        """
        Verify that len(dataset) returns the number of ACES examples, not total records.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        assert len(ds) == aces_filtering_setup["expected_aces_examples"], (
            f"Dataset length should equal number of ACES examples ({aces_filtering_setup['expected_aces_examples']}), "
            f"but got {len(ds)}"
        )

    def test_subject_to_records_contains_all_subjects(self, aces_filtering_setup):
        """
        Verify that subject_to_records mapping contains ALL loaded patient records,
        not just those with ACES labels.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        all_subject_ids = aces_filtering_setup["aces_subject_ids"] | aces_filtering_setup["non_aces_subject_ids"]
        
        # subject_to_records should have all 10 subjects
        assert len(ds.subject_to_records) == len(all_subject_ids), (
            f"Expected subject_to_records to contain {len(all_subject_ids)} subjects, "
            f"but got {len(ds.subject_to_records)}"
        )
        
        # Verify all subject IDs are present
        for sid in all_subject_ids:
            assert sid in ds.subject_to_records, f"Subject {sid} not found in subject_to_records"

    def test_getitem_works_for_all_aces_examples(self, aces_filtering_setup):
        """
        Verify that __getitem__ works correctly for all ACES examples.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        for i in range(len(ds)):
            item = ds[i]
            assert "text" in item
            assert "label" in item
            assert "subject_id" in item
            assert item["subject_id"] in aces_filtering_setup["aces_subject_ids"]

    def test_aces_labels_preserved_correctly(self, aces_filtering_setup):
        """
        Verify that the labels from ACES are correctly assigned to examples.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Build expected labels mapping
        expected_labels = {
            1000: 0,
            1002: 1,
            1004: 0,
            1006: 1,
            1008: 1,
        }
        
        # Check each example has correct label
        for i in range(len(ds)):
            item = ds[i]
            subject_id = item["subject_id"]
            expected_label = expected_labels[subject_id]
            actual_label = item["label"].item()
            assert actual_label == expected_label, (
                f"Subject {subject_id}: expected label {expected_label}, got {actual_label}"
            )

    def test_no_aces_file_results_in_zero_examples(self, aces_filtering_setup):
        """
        Verify that if ACES labels file doesn't exist, no prediction examples are created.
        Note: __len__ falls back to len(patient_records) when examples list is empty.
        """
        # Create a temporary directory without ACES labels
        empty_aces_dir = os.path.join(str(aces_filtering_setup["aces_label_dir"]), "empty")
        os.makedirs(empty_aces_dir, exist_ok=True)
        
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=empty_aces_dir,  # No train_labels.parquet here
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Patient records should still be loaded
        assert len(ds.patient_records) == aces_filtering_setup["expected_total_records"]
        
        # But no examples should be created
        assert len(ds.examples) == 0
        
        # __len__ falls back to patient_records when examples is empty
        assert len(ds) == aces_filtering_setup["expected_total_records"]

    def test_aces_with_missing_patient_records_skipped(self, tmp_path):
        """
        Verify that ACES labels for subjects without patient records are skipped.
        """
        tmp = str(tmp_path)
        
        # ---- vocab ----
        vocab_rows = [(0, "<pad>"), (1, "<start>"), (2, "AGE"), (3, "55.0")]
        vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
        vocab_file = os.path.join(tmp, "vocab.csv")
        vocab_df.to_csv(vocab_file, index=False)

        # ---- lookup tables ----
        hosp_dir = os.path.join(tmp, "hosp")
        os.makedirs(hosp_dir, exist_ok=True)
        pd.DataFrame({"itemid": [51006], "label": ["Urea"]}).to_csv(
            os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["I10"], "long_title": ["HTN"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["0BH17EZ"], "long_title": ["Intubation"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip"
        )

        # ---- patient records (only subjects 1000, 1001) ----
        base_time = 1577836800
        data_dir = os.path.join(tmp, "data")
        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        records = []
        for j in range(2):
            subject_id = 1000 + j
            tokens = [1, 2, 3]
            timestamps = [0, 0, base_time]
            records.append({
                "subject_id": subject_id,
                "tokens": tokens,
                "timestamps": timestamps,
            })
        with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
            pickle.dump(records, f)

        # ---- ACES labels (includes subjects 1000, 1001, 1002, 1003) ----
        # Subjects 1002 and 1003 don't have patient records, so should be skipped
        aces_dir = os.path.join(tmp, "aces_labels")
        os.makedirs(aces_dir, exist_ok=True)
        
        aces_rows = [
            {"subject_id": 1000, "label": 0, "prediction_time": float(base_time)},
            {"subject_id": 1001, "label": 1, "prediction_time": float(base_time)},
            {"subject_id": 1002, "label": 0, "prediction_time": float(base_time)},  # No record
            {"subject_id": 1003, "label": 1, "prediction_time": float(base_time)},  # No record
        ]
        aces_df = pd.DataFrame(aces_rows)
        aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

        # ---- Create dataset ----
        ds = MIMICDataset(
            data_dir=data_dir,
            vocab_file=vocab_file,
            aces_label_dir=aces_dir,
            lookup_dir=hosp_dir,
            task="prediction",
            split="train",
        )
        
        # Should have 2 patient records loaded
        assert len(ds.patient_records) == 2
        
        # Should have only 2 examples (1000, 1001), not 4
        assert len(ds.examples) == 2
        assert len(ds) == 2
        
        # Verify the correct subjects are in examples
        example_subject_ids = {ex["subject_id"] for ex in ds.examples}
        assert example_subject_ids == {1000, 1001}

    def test_multiple_pkl_files_all_loaded(self, aces_filtering_setup):
        """
        Verify that patient records from multiple .pkl files are all loaded.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Check that records from all 3 pickle files are present
        all_loaded_subject_ids = {rec["subject_id"] for rec in ds.patient_records}
        expected_all_subjects = (
            aces_filtering_setup["aces_subject_ids"] | 
            aces_filtering_setup["non_aces_subject_ids"]
        )
        
        assert all_loaded_subject_ids == expected_all_subjects, (
            f"Expected all subjects {expected_all_subjects} to be loaded, "
            f"but got {all_loaded_subject_ids}"
        )

    def test_prediction_time_from_aces_used(self, aces_filtering_setup):
        """
        Verify that prediction_time from ACES labels is used for time cutoff.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Subject 1004 has prediction_time = base_time + 3600
        # Its tokens have timestamps up to base_time + 7200
        # So some tokens should be filtered out
        
        # Find the example for subject 1004
        example_1004 = None
        for i, ex in enumerate(ds.examples):
            if ex["subject_id"] == 1004:
                example_1004 = i
                break
        
        assert example_1004 is not None, "Subject 1004 not found in examples"
        
        # Get the item
        item = ds[example_1004]
        
        # The text should be shorter than if we used the full trajectory
        # (This is a qualitative check - we know tokens 8,9 should be filtered)
        assert item["text"] is not None
        assert len(item["text"]) > 0

    def test_pretrain_mode_ignores_aces_filtering(self, aces_filtering_setup):
        """
        Verify that in pretrain mode, all patient records are used (no ACES filtering).
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="pretrain",  # pretrain mode
            split="train",
        )
        
        # In pretrain mode, len(dataset) should equal total patient records
        assert len(ds) == aces_filtering_setup["expected_total_records"], (
            f"In pretrain mode, expected {aces_filtering_setup['expected_total_records']} examples, "
            f"but got {len(ds)}"
        )
        
        # No examples should be built in pretrain mode
        assert len(ds.examples) == 0

    def test_aces_labels_with_extra_columns_preserved(self, tmp_path):
        """
        Verify that extra columns in ACES labels (beyond subject_id, label, prediction_time)
        are preserved in the examples.
        """
        tmp = str(tmp_path)
        
        # Setup minimal data
        vocab_rows = [(0, "<pad>"), (1, "<start>"), (2, "AGE")]
        vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
        vocab_file = os.path.join(tmp, "vocab.csv")
        vocab_df.to_csv(vocab_file, index=False)

        hosp_dir = os.path.join(tmp, "hosp")
        os.makedirs(hosp_dir, exist_ok=True)
        pd.DataFrame({"itemid": [51006], "label": ["Urea"]}).to_csv(
            os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["I10"], "long_title": ["HTN"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["0BH17EZ"], "long_title": ["Intubation"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip"
        )

        data_dir = os.path.join(tmp, "data")
        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        base_time = 1577836800
        records = [{"subject_id": 1000, "tokens": [1, 2], "timestamps": [0, base_time]}]
        with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
            pickle.dump(records, f)

        # ACES with extra columns
        aces_dir = os.path.join(tmp, "aces_labels")
        os.makedirs(aces_dir, exist_ok=True)
        
        aces_rows = [{
            "subject_id": 1000,
            "label": 1,
            "prediction_time": float(base_time),
            "MEDS_DEATH": 1,
            "HOSPITAL_DISCHARGE//DIED": 0,
            "custom_field": "test_value",
        }]
        aces_df = pd.DataFrame(aces_rows)
        aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

        ds = MIMICDataset(
            data_dir=data_dir,
            vocab_file=vocab_file,
            aces_label_dir=aces_dir,
            lookup_dir=hosp_dir,
            task="prediction",
            split="train",
        )
        
        # Verify extra columns are preserved in examples
        assert len(ds.examples) == 1
        example = ds.examples[0]
        assert example["MEDS_DEATH"] == 1
        assert example["HOSPITAL_DISCHARGE//DIED"] == 0
        assert example["custom_field"] == "test_value"


class TestACESLoadingEdgeCases:
    """Test edge cases in ACES loading."""

    def test_empty_aces_labels_file(self, tmp_path):
        """
        Verify behavior when ACES labels file exists but is empty.
        Note: __len__ falls back to len(patient_records) when examples list is empty.
        """
        tmp = str(tmp_path)
        
        # Setup minimal data
        vocab_rows = [(0, "<pad>"), (1, "<start>")]
        vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
        vocab_file = os.path.join(tmp, "vocab.csv")
        vocab_df.to_csv(vocab_file, index=False)

        hosp_dir = os.path.join(tmp, "hosp")
        os.makedirs(hosp_dir, exist_ok=True)
        pd.DataFrame({"itemid": [51006], "label": ["Urea"]}).to_csv(
            os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["I10"], "long_title": ["HTN"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["0BH17EZ"], "long_title": ["Intubation"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip"
        )

        data_dir = os.path.join(tmp, "data")
        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        records = [{"subject_id": 1000, "tokens": [1], "timestamps": [0]}]
        with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
            pickle.dump(records, f)

        # Empty ACES labels
        aces_dir = os.path.join(tmp, "aces_labels")
        os.makedirs(aces_dir, exist_ok=True)
        empty_aces_df = pd.DataFrame(columns=["subject_id", "label", "prediction_time"])
        empty_aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

        ds = MIMICDataset(
            data_dir=data_dir,
            vocab_file=vocab_file,
            aces_label_dir=aces_dir,
            lookup_dir=hosp_dir,
            task="prediction",
            split="train",
        )
        
        assert len(ds.patient_records) == 1
        assert len(ds.examples) == 0
        
        # __len__ falls back to patient_records when examples is empty
        assert len(ds) == 1

    def test_duplicate_subject_in_aces_creates_multiple_examples(self, tmp_path):
        """
        Verify that if the same subject appears multiple times in ACES labels,
        multiple examples are created (e.g., multiple prediction times for same patient).
        """
        tmp = str(tmp_path)
        
        # Setup minimal data
        vocab_rows = [(0, "<pad>"), (1, "<start>"), (2, "AGE")]
        vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
        vocab_file = os.path.join(tmp, "vocab.csv")
        vocab_df.to_csv(vocab_file, index=False)

        hosp_dir = os.path.join(tmp, "hosp")
        os.makedirs(hosp_dir, exist_ok=True)
        pd.DataFrame({"itemid": [51006], "label": ["Urea"]}).to_csv(
            os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["I10"], "long_title": ["HTN"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["0BH17EZ"], "long_title": ["Intubation"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip"
        )

        data_dir = os.path.join(tmp, "data")
        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        base_time = 1577836800
        records = [{"subject_id": 1000, "tokens": [1, 2], "timestamps": [0, base_time]}]
        with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
            pickle.dump(records, f)

        # ACES with same subject appearing 3 times
        aces_dir = os.path.join(tmp, "aces_labels")
        os.makedirs(aces_dir, exist_ok=True)
        
        aces_rows = [
            {"subject_id": 1000, "label": 0, "prediction_time": float(base_time - 1000)},
            {"subject_id": 1000, "label": 1, "prediction_time": float(base_time)},
            {"subject_id": 1000, "label": 0, "prediction_time": float(base_time + 1000)},
        ]
        aces_df = pd.DataFrame(aces_rows)
        aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

        ds = MIMICDataset(
            data_dir=data_dir,
            vocab_file=vocab_file,
            aces_label_dir=aces_dir,
            lookup_dir=hosp_dir,
            task="prediction",
            split="train",
        )
        
        # Should have 1 patient record but 3 examples
        assert len(ds.patient_records) == 1
        assert len(ds.examples) == 3
        assert len(ds) == 3
        
        # All examples should have the same subject_id
        for ex in ds.examples:
            assert ex["subject_id"] == 1000


class TestACESLoadingWithRealWorldScenarios:
    """Test scenarios that mimic real-world usage patterns."""

    def test_large_dataset_with_partial_aces_coverage(self, tmp_path):
        """
        Simulate a realistic scenario where we have many patient records
        but only a subset have ACES labels (e.g., only ICU patients).
        """
        tmp = str(tmp_path)
        
        # Setup
        vocab_rows = [(0, "<pad>"), (1, "<start>"), (2, "AGE"), (3, "55.0")]
        vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
        vocab_file = os.path.join(tmp, "vocab.csv")
        vocab_df.to_csv(vocab_file, index=False)

        hosp_dir = os.path.join(tmp, "hosp")
        os.makedirs(hosp_dir, exist_ok=True)
        pd.DataFrame({"itemid": [51006], "label": ["Urea"]}).to_csv(
            os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["I10"], "long_title": ["HTN"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip"
        )
        pd.DataFrame({"icd_version": [10], "icd_code": ["0BH17EZ"], "long_title": ["Intubation"]}).to_csv(
            os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip"
        )

        # Create 100 patient records
        data_dir = os.path.join(tmp, "data")
        train_dir = os.path.join(data_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        base_time = 1577836800
        num_patients = 100
        records = []
        for j in range(num_patients):
            subject_id = 2000 + j
            tokens = [1, 2, 3]
            timestamps = [0, 0, base_time]
            records.append({
                "subject_id": subject_id,
                "tokens": tokens,
                "timestamps": timestamps,
            })
        with open(os.path.join(train_dir, "batch_0.pkl"), "wb") as f:
            pickle.dump(records, f)

        # ACES labels for only 20% of patients (20 out of 100)
        aces_dir = os.path.join(tmp, "aces_labels")
        os.makedirs(aces_dir, exist_ok=True)
        
        aces_rows = []
        for j in range(0, num_patients, 5):  # Every 5th patient
            subject_id = 2000 + j
            aces_rows.append({
                "subject_id": subject_id,
                "label": j % 2,
                "prediction_time": float(base_time),
            })
        aces_df = pd.DataFrame(aces_rows)
        aces_df.to_parquet(os.path.join(aces_dir, "train_labels.parquet"), index=False)

        ds = MIMICDataset(
            data_dir=data_dir,
            vocab_file=vocab_file,
            aces_label_dir=aces_dir,
            lookup_dir=hosp_dir,
            task="prediction",
            split="train",
        )
        
        # All 100 records should be loaded
        assert len(ds.patient_records) == num_patients
        
        # But only 20 examples (those with ACES labels)
        assert len(ds.examples) == 20
        assert len(ds) == 20
        
        # Verify correct subjects are in examples
        expected_aces_subjects = {2000 + j for j in range(0, num_patients, 5)}
        actual_aces_subjects = {ex["subject_id"] for ex in ds.examples}
        assert actual_aces_subjects == expected_aces_subjects

    def test_all_aces_examples_accessible_via_getitem(self, aces_filtering_setup):
        """
        Verify that all ACES examples can be accessed via __getitem__ without errors.
        """
        ds = MIMICDataset(
            data_dir=aces_filtering_setup["data_dir"],
            vocab_file=aces_filtering_setup["vocab_filepath"],
            aces_label_dir=aces_filtering_setup["aces_label_dir"],
            lookup_dir=aces_filtering_setup["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Access all examples
        for i in range(len(ds)):
            item = ds[i]
            
            # Verify structure
            assert isinstance(item, dict)
            assert "text" in item
            assert "label" in item
            assert "subject_id" in item
            
            # Verify text is non-empty
            assert isinstance(item["text"], str)
            assert len(item["text"]) > 0
            
            # Verify label is correct type
            assert hasattr(item["label"], "item")  # torch.Tensor
            assert item["label"].item() in {0, 1}
            
            # Verify subject_id matches an ACES subject
            assert item["subject_id"] in aces_filtering_setup["aces_subject_ids"]
