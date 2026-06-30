"""
Tests for dataset initialization from config files.

These tests verify that:
1. Datasets can be created from config files
2. Dataset parameters are correctly passed from config
3. Task mode is correctly set based on config
4. ACES labels are loaded when task='prediction'
5. Dataset factory creates the correct dataset type
"""
import os
import pytest
import yaml
from pathlib import Path

from src.data.dataset_factory import get_dataset
from src.data.mimic_dataset import MIMICDataset
from src.data.cprd_dataset import CPRDDataset


@pytest.fixture
def mimic_classifier_config():
    """Load the MIMIC classifier config."""
    config_path = Path(__file__).parent.parent / "src" / "configs" / "mimic_classifier.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestDatasetFactoryFromConfig:
    """Test that dataset_factory correctly creates datasets from config."""

    def test_factory_creates_mimic_dataset(self, mimic_classifier_config):
        """Verify that factory creates MIMICDataset for MIMIC config."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'  # Use tuning split (smaller, faster to load)
        )
        
        assert isinstance(dataset, MIMICDataset), (
            f"Expected MIMICDataset, got {type(dataset)}"
        )

    def test_factory_uses_task_from_config(self, mimic_classifier_config):
        """Verify that factory passes task parameter from config to dataset."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        # Verify task was set correctly
        assert dataset.task == data_config.get('task', 'pretrain'), (
            f"Dataset task '{dataset.task}' doesn't match config task '{data_config.get('task')}'"
        )

    def test_factory_passes_label_column(self, mimic_classifier_config):
        """Verify that factory passes label_column from config."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        expected_label_column = data_config.get('label_column', 'label')
        assert dataset.label_column == expected_label_column, (
            f"Dataset label_column '{dataset.label_column}' doesn't match "
            f"config '{expected_label_column}'"
        )


class TestMIMICDatasetInitialization:
    """Test MIMIC dataset initialization with real config."""

    def test_mimic_dataset_loads_with_prediction_task(self, mimic_classifier_config):
        """Verify that MIMIC dataset loads correctly with task='prediction'."""
        data_config = mimic_classifier_config['data']
        
        # Ensure task is set to prediction
        assert data_config.get('task') == 'prediction', (
            "This test requires task='prediction' in config"
        )
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        # Verify ACES examples were created
        assert len(dataset.examples) > 0, (
            "No prediction examples created! ACES labels may not be loading."
        )
        
        # Verify subject_to_records was built
        assert len(dataset.subject_to_records) > 0, (
            "subject_to_records is empty! Patient records may not be loading."
        )

    def test_mimic_dataset_loads_patient_records(self, mimic_classifier_config):
        """Verify that patient records are loaded from data_dir."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        assert len(dataset.patient_records) > 0, (
            "No patient records loaded from data_dir"
        )

    def test_mimic_dataset_has_both_classes_in_prediction_mode(self, mimic_classifier_config):
        """
        CRITICAL: Verify that dataset has both positive and negative examples.
        
        If only one class is present, it indicates:
        - ACES labels are not loading
        - Task mode is wrong (pretrain instead of prediction)
        - All labels are defaulting to 0
        """
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        # Sample labels from dataset
        labels = []
        sample_size = min(100, len(dataset))
        for i in range(sample_size):
            item = dataset[i]
            labels.append(item['label'].item())
        
        unique_labels = set(labels)
        assert len(unique_labels) > 1, (
            f"Only one class present in dataset: {unique_labels}\n"
            f"This indicates ACES labels are not loading correctly!\n"
            f"Check:\n"
            f"  1. task='prediction' is set in config\n"
            f"  2. aces_label_dir points to correct directory with label files\n"
            f"  3. Label files exist and have multiple classes"
        )
        
        # For binary classification, should have classes 0 and 1
        assert 0 in unique_labels and 1 in unique_labels, (
            f"Expected classes 0 and 1 for binary classification, got: {unique_labels}"
        )

    def test_mimic_dataset_length_matches_aces_examples(self, mimic_classifier_config):
        """Verify that dataset length equals number of ACES examples in prediction mode."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        if data_config.get('task') == 'prediction':
            assert len(dataset) == len(dataset.examples), (
                f"Dataset length ({len(dataset)}) should equal number of ACES examples "
                f"({len(dataset.examples)}) in prediction mode"
            )

    def test_mimic_examples_less_than_total_records(self, mimic_classifier_config):
        """
        Verify that prediction examples < total patient records.
        
        This is expected because not all patients have ACES labels
        (e.g., only ICU patients for ICU mortality task).
        """
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        if data_config.get('task') == 'prediction':
            assert len(dataset.examples) < len(dataset.patient_records), (
                f"Expected fewer ACES examples ({len(dataset.examples)}) than "
                f"total patient records ({len(dataset.patient_records)}), "
                f"but they are equal or examples > records. This suggests filtering is not working."
            )


class TestDatasetLoadingAcrossSplits:
    """Test that datasets load correctly for all splits."""

    @pytest.mark.parametrize("split", ["train", "tuning", "held_out"])
    def test_dataset_loads_for_all_splits(self, mimic_classifier_config, split):
        """Verify that dataset can be loaded for each split."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split=split
        )
        
        assert dataset is not None
        assert len(dataset) > 0, f"Dataset for split '{split}' is empty"

    @pytest.mark.parametrize("split", ["train", "tuning", "held_out"])
    def test_all_splits_have_both_classes(self, mimic_classifier_config, split):
        """
        Verify that all splits have both positive and negative examples.
        
        This is critical for proper training and evaluation.
        """
        data_config = mimic_classifier_config['data']
        
        # Skip if not prediction task
        if data_config.get('task') != 'prediction':
            pytest.skip("Test only applicable for prediction task")
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split=split
        )
        
        # Sample labels
        labels = []
        sample_size = min(50, len(dataset))
        for i in range(sample_size):
            item = dataset[i]
            labels.append(item['label'].item())
        
        unique_labels = set(labels)
        assert len(unique_labels) > 1, (
            f"Split '{split}' has only one class: {unique_labels}\n"
            f"Both classes must be present for proper training/evaluation."
        )

    def test_train_split_larger_than_validation(self, mimic_classifier_config):
        """Verify that train split is larger than validation split."""
        data_config = mimic_classifier_config['data']
        
        train_dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='train'
        )
        
        val_dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        assert len(train_dataset) > len(val_dataset), (
            f"Train split ({len(train_dataset)}) should be larger than "
            f"validation split ({len(val_dataset)})"
        )


class TestDatasetItemStructure:
    """Test that dataset items have the correct structure."""

    def test_dataset_item_has_required_keys(self, mimic_classifier_config):
        """Verify that dataset items have all required keys."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        item = dataset[0]
        required_keys = ['text', 'label', 'subject_id']
        
        for key in required_keys:
            assert key in item, f"Dataset item missing required key: {key}"

    def test_dataset_text_is_nonempty(self, mimic_classifier_config):
        """Verify that text field is non-empty string."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert isinstance(item['text'], str), (
                f"Item {i}: text should be string, got {type(item['text'])}"
            )
            assert len(item['text']) > 0, (
                f"Item {i}: text should not be empty"
            )

    def test_dataset_label_is_tensor(self, mimic_classifier_config):
        """Verify that label is a torch tensor."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        import torch
        item = dataset[0]
        assert isinstance(item['label'], torch.Tensor), (
            f"Label should be torch.Tensor, got {type(item['label'])}"
        )
        assert item['label'].dtype == torch.long, (
            f"Label should be torch.long, got {item['label'].dtype}"
        )

    def test_dataset_labels_in_valid_range(self, mimic_classifier_config):
        """Verify that labels are in valid range [0, num_labels-1]."""
        data_config = mimic_classifier_config['data']
        model_config = mimic_classifier_config['model']
        num_labels = model_config['num_labels']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        for i in range(min(50, len(dataset))):
            item = dataset[i]
            label_value = item['label'].item()
            assert 0 <= label_value < num_labels, (
                f"Item {i}: label {label_value} out of range [0, {num_labels-1}]"
            )


class TestPredictionVsPretrainMode:
    """Test differences between prediction and pretrain modes."""

    def test_prediction_mode_creates_examples(self, tmp_path):
        """Verify that prediction mode creates ACES examples."""
        # Use the test fixture to create a minimal dataset
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=5)
        
        dataset = MIMICDataset(
            data_dir=paths["data_dir"],
            vocab_file=paths["vocab_filepath"],
            aces_label_dir=paths["aces_label_dir"],
            lookup_dir=paths["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        assert len(dataset.examples) > 0, (
            "Prediction mode should create examples from ACES labels"
        )
        assert len(dataset.subject_to_records) > 0, (
            "Prediction mode should build subject_to_records mapping"
        )

    def test_pretrain_mode_no_examples(self, tmp_path):
        """Verify that pretrain mode does not create ACES examples."""
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=5)
        
        dataset = MIMICDataset(
            data_dir=paths["data_dir"],
            vocab_file=paths["vocab_filepath"],
            aces_label_dir=paths["aces_label_dir"],
            lookup_dir=paths["lookup_dir"],
            task="pretrain",
            split="train",
        )
        
        assert len(dataset.examples) == 0, (
            "Pretrain mode should not create examples"
        )
        assert len(dataset) == len(dataset.patient_records), (
            "Pretrain mode length should equal patient records"
        )

    def test_pretrain_mode_all_labels_zero(self, tmp_path):
        """Verify that pretrain mode returns label=0 for all examples."""
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=5)
        
        dataset = MIMICDataset(
            data_dir=paths["data_dir"],
            vocab_file=paths["vocab_filepath"],
            aces_label_dir=paths["aces_label_dir"],
            lookup_dir=paths["lookup_dir"],
            task="pretrain",
            split="train",
        )
        
        # All labels should be 0 in pretrain mode
        for i in range(len(dataset)):
            item = dataset[i]
            assert item['label'].item() == 0, (
                f"Pretrain mode should return label=0, got {item['label'].item()}"
            )

    def test_prediction_mode_has_varied_labels(self, tmp_path):
        """Verify that prediction mode returns varied labels from ACES."""
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=5)
        
        dataset = MIMICDataset(
            data_dir=paths["data_dir"],
            vocab_file=paths["vocab_filepath"],
            aces_label_dir=paths["aces_label_dir"],
            lookup_dir=paths["lookup_dir"],
            task="prediction",
            split="train",
        )
        
        # Get all labels
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        unique_labels = set(labels)
        
        assert len(unique_labels) > 1, (
            f"Prediction mode should have multiple classes, got only: {unique_labels}"
        )


class TestDatasetFactoryDefaults:
    """Test that dataset_factory handles missing config fields correctly."""

    def test_factory_defaults_to_pretrain_if_task_missing(self, tmp_path):
        """
        Verify that factory defaults to 'pretrain' if task is not in config.
        
        This is the bug that caused the original issue!
        """
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=3)
        
        # Create config without 'task' field
        data_config = {
            'data_dir': paths["data_dir"],
            'vocab_filepath': paths["vocab_filepath"],
            'aces_label_dir': paths["aces_label_dir"],
            'lookup_dir': paths["lookup_dir"],
        }
        
        dataset = get_dataset(
            dataset_name='MIMIC',
            data_config=data_config,
            split='train'
        )
        
        # Should default to pretrain
        assert dataset.task == 'pretrain', (
            "Factory should default to 'pretrain' when task is not specified"
        )
        
        # Should not create ACES examples
        assert len(dataset.examples) == 0, (
            "Pretrain mode (default) should not create ACES examples"
        )

    def test_factory_defaults_to_label_column(self, tmp_path):
        """Verify that factory defaults to 'label' for label_column if not specified."""
        from tests.conftest import write_fake_mimic_data
        
        paths = write_fake_mimic_data(str(tmp_path), records_per_split=3)
        
        # Create config without 'label_column' field
        data_config = {
            'data_dir': paths["data_dir"],
            'vocab_filepath': paths["vocab_filepath"],
            'aces_label_dir': paths["aces_label_dir"],
            'lookup_dir': paths["lookup_dir"],
            'task': 'prediction',
        }
        
        dataset = get_dataset(
            dataset_name='MIMIC',
            data_config=data_config,
            split='train'
        )
        
        # Should default to 'label'
        assert dataset.label_column == 'label', (
            "Factory should default to 'label' for label_column"
        )


class TestRealWorldDatasetLoading:
    """Test dataset loading with real config and real data paths."""

    def test_train_dataset_has_many_examples(self, mimic_classifier_config):
        """Verify that train dataset has a substantial number of examples."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='train'
        )
        
        # Train set should have thousands of examples for MIMIC ICU mortality
        assert len(dataset) > 1000, (
            f"Train dataset seems too small: {len(dataset)} examples. "
            f"Expected > 1000 for MIMIC ICU mortality task."
        )

    def test_class_imbalance_is_reasonable(self, mimic_classifier_config):
        """
        Verify that class distribution is imbalanced but not extreme.
        
        For ICU mortality, we expect:
        - Minority class (deaths): 5-15% of examples
        - Majority class (survivors): 85-95% of examples
        """
        data_config = mimic_classifier_config['data']
        
        if data_config.get('task') != 'prediction':
            pytest.skip("Test only applicable for prediction task")
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        # Sample labels
        labels = []
        sample_size = min(500, len(dataset))
        for i in range(sample_size):
            item = dataset[i]
            labels.append(item['label'].item())
        
        # Calculate positive rate
        positive_rate = sum(labels) / len(labels)
        
        assert 0.01 < positive_rate < 0.50, (
            f"Class distribution seems unusual. Positive rate: {positive_rate:.2%}\n"
            f"Expected between 1% and 50% for typical medical prediction tasks."
        )

    def test_all_examples_have_valid_subject_ids(self, mimic_classifier_config):
        """Verify that all examples have valid subject IDs."""
        data_config = mimic_classifier_config['data']
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        for i in range(min(20, len(dataset))):
            item = dataset[i]
            subject_id = item['subject_id']
            
            assert subject_id is not None, f"Item {i} has None subject_id"
            assert isinstance(subject_id, (int, str)), (
                f"Item {i}: subject_id should be int or str, got {type(subject_id)}"
            )

    def test_subject_ids_match_between_examples_and_records(self, mimic_classifier_config):
        """
        Verify that subject_ids in examples exist in patient records.
        
        This ensures the ACES filtering is working correctly.
        """
        data_config = mimic_classifier_config['data']
        
        if data_config.get('task') != 'prediction':
            pytest.skip("Test only applicable for prediction task")
        
        dataset = get_dataset(
            dataset_name=data_config['dataset'],
            data_config=data_config,
            split='tuning'
        )
        
        # Get subject IDs from examples
        example_subject_ids = {ex['subject_id'] for ex in dataset.examples}
        
        # Get subject IDs from subject_to_records
        record_subject_ids = set(dataset.subject_to_records.keys())
        
        # All example subject IDs should be in records
        assert example_subject_ids.issubset(record_subject_ids), (
            f"Found {len(example_subject_ids - record_subject_ids)} example subject IDs "
            f"that don't exist in patient records"
        )
