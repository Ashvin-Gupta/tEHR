"""
Tests for configuration loading and validation.

These tests verify that:
1. Config files can be loaded successfully
2. Required fields are present in configs
3. Config values are of the correct type
4. Dataset-specific configs have the right parameters
5. Task parameter is set correctly for classification vs pretrain
"""
import os
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config_dir():
    """Return the path to the configs directory."""
    return Path(__file__).parent.parent / "src" / "configs"


@pytest.fixture
def mimic_classifier_config(config_dir):
    """Load the MIMIC classifier config."""
    config_path = config_dir / "mimic_classifier.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestConfigStructure:
    """Test that config files have the correct structure."""

    def test_config_file_exists(self, config_dir):
        """Verify that the MIMIC classifier config file exists."""
        config_path = config_dir / "mimic_classifier.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_has_required_top_level_keys(self, mimic_classifier_config):
        """Verify that config has all required top-level sections."""
        required_keys = ['name', 'model', 'training', 'data']
        for key in required_keys:
            assert key in mimic_classifier_config, f"Missing required key: {key}"

    def test_model_section_has_required_fields(self, mimic_classifier_config):
        """Verify that model section has all required fields."""
        model_config = mimic_classifier_config['model']
        required_fields = [
            'pretrained_checkpoint',
            'max_length',
            'hidden_size',
            'num_labels',
            'freeze_llm',
            'train_lora',
        ]
        for field in required_fields:
            assert field in model_config, f"Missing required model field: {field}"

    def test_training_section_has_required_fields(self, mimic_classifier_config):
        """Verify that training section has all required fields."""
        training_config = mimic_classifier_config['training']
        required_fields = [
            'output_dir',
            'epochs',
            'batch_size',
            'eval_batch_size',
            'learning_rate',
            'logging_steps',
            'eval_steps',
        ]
        for field in required_fields:
            assert field in training_config, f"Missing required training field: {field}"

    def test_data_section_has_required_fields(self, mimic_classifier_config):
        """Verify that data section has all required fields."""
        data_config = mimic_classifier_config['data']
        required_fields = [
            'data_dir',
            'vocab_filepath',
            'dataset',
        ]
        for field in required_fields:
            assert field in data_config, f"Missing required data field: {field}"


class TestMIMICConfigSpecific:
    """Test MIMIC-specific configuration requirements."""

    def test_mimic_dataset_type(self, mimic_classifier_config):
        """Verify that dataset is set to MIMIC."""
        assert mimic_classifier_config['data']['dataset'] == 'MIMIC'

    def test_mimic_has_aces_label_dir(self, mimic_classifier_config):
        """Verify that MIMIC config has aces_label_dir field."""
        data_config = mimic_classifier_config['data']
        assert 'aces_label_dir' in data_config, (
            "MIMIC config must have 'aces_label_dir' for loading prediction labels"
        )

    def test_mimic_has_lookup_dir(self, mimic_classifier_config):
        """Verify that MIMIC config has lookup_dir field."""
        data_config = mimic_classifier_config['data']
        assert 'lookup_dir' in data_config, (
            "MIMIC config must have 'lookup_dir' for token translation"
        )

    def test_mimic_has_task_field_for_classification(self, mimic_classifier_config):
        """
        CRITICAL: Verify that task='prediction' is set for classification.
        Without this, the dataset defaults to 'pretrain' mode and ACES labels are not loaded.
        """
        data_config = mimic_classifier_config['data']
        assert 'task' in data_config, (
            "MIMIC config MUST have 'task' field. "
            "Without it, dataset defaults to 'pretrain' mode and ACES labels are not loaded!"
        )
        assert data_config['task'] == 'prediction', (
            f"For classification, task must be 'prediction', got '{data_config['task']}'"
        )

    def test_num_labels_is_two_for_binary(self, mimic_classifier_config):
        """Verify that num_labels=2 for binary classification."""
        model_config = mimic_classifier_config['model']
        multi_label = mimic_classifier_config['training'].get('multi_label', False)
        
        if not multi_label:
            assert model_config['num_labels'] == 2, (
                "For binary classification, num_labels should be 2"
            )


class TestConfigValueTypes:
    """Test that config values have the correct types."""

    def test_numeric_fields_are_numeric(self, mimic_classifier_config):
        """Verify that numeric fields have numeric values."""
        model_config = mimic_classifier_config['model']
        training_config = mimic_classifier_config['training']
        
        # Integer fields
        assert isinstance(model_config['max_length'], int)
        assert isinstance(model_config['hidden_size'], int)
        assert isinstance(model_config['num_labels'], int)
        assert isinstance(training_config['epochs'], int)
        assert isinstance(training_config['batch_size'], int)
        assert isinstance(training_config['eval_batch_size'], int)
        
        # Float fields (can be string like "1e-6" or float)
        lr = training_config['learning_rate']
        assert isinstance(lr, (int, float, str)), "learning_rate must be numeric or string"
        if isinstance(lr, str):
            float(lr)  # Should not raise

    def test_boolean_fields_are_boolean(self, mimic_classifier_config):
        """Verify that boolean fields have boolean values."""
        model_config = mimic_classifier_config['model']
        training_config = mimic_classifier_config['training']
        
        assert isinstance(model_config['freeze_llm'], bool)
        assert isinstance(model_config['train_lora'], bool)
        assert isinstance(training_config['multi_label'], bool)
        assert isinstance(training_config.get('fp16', False), bool)
        assert isinstance(training_config.get('bf16', True), bool)

    def test_path_fields_are_strings(self, mimic_classifier_config):
        """Verify that path fields are strings."""
        data_config = mimic_classifier_config['data']
        model_config = mimic_classifier_config['model']
        training_config = mimic_classifier_config['training']
        
        assert isinstance(data_config['data_dir'], str)
        assert isinstance(data_config['vocab_filepath'], str)
        assert isinstance(data_config['aces_label_dir'], str)
        assert isinstance(data_config['lookup_dir'], str)
        assert isinstance(model_config['pretrained_checkpoint'], str)
        assert isinstance(training_config['output_dir'], str)


class TestConfigPathsExist:
    """Test that paths in config point to existing locations (when applicable)."""

    def test_data_dir_exists(self, mimic_classifier_config):
        """Verify that data_dir exists."""
        data_dir = mimic_classifier_config['data']['data_dir']
        assert os.path.exists(data_dir), f"data_dir does not exist: {data_dir}"

    def test_vocab_file_exists(self, mimic_classifier_config):
        """Verify that vocab file exists."""
        vocab_file = mimic_classifier_config['data']['vocab_filepath']
        assert os.path.exists(vocab_file), f"vocab_filepath does not exist: {vocab_file}"

    def test_aces_label_dir_exists(self, mimic_classifier_config):
        """Verify that ACES label directory exists."""
        aces_dir = mimic_classifier_config['data']['aces_label_dir']
        assert os.path.exists(aces_dir), f"aces_label_dir does not exist: {aces_dir}"

    def test_lookup_dir_exists(self, mimic_classifier_config):
        """Verify that lookup directory exists."""
        lookup_dir = mimic_classifier_config['data']['lookup_dir']
        assert os.path.exists(lookup_dir), f"lookup_dir does not exist: {lookup_dir}"

    def test_aces_label_files_exist(self, mimic_classifier_config):
        """Verify that ACES label files exist for all splits."""
        aces_dir = mimic_classifier_config['data']['aces_label_dir']
        
        for split in ['train', 'tuning', 'held_out']:
            label_file = os.path.join(aces_dir, f"{split}_labels.parquet")
            assert os.path.exists(label_file), (
                f"ACES labels file missing for split '{split}': {label_file}"
            )

    def test_pretrained_checkpoint_exists(self, mimic_classifier_config):
        """Verify that pretrained checkpoint exists."""
        checkpoint = mimic_classifier_config['model']['pretrained_checkpoint']
        assert os.path.exists(checkpoint), (
            f"pretrained_checkpoint does not exist: {checkpoint}"
        )


class TestConfigConsistency:
    """Test that config values are consistent with each other."""

    def test_multi_label_consistent_with_num_labels(self, mimic_classifier_config):
        """Verify that multi_label setting is consistent with num_labels."""
        model_config = mimic_classifier_config['model']
        training_config = mimic_classifier_config['training']
        
        multi_label = training_config.get('multi_label', False)
        num_labels = model_config['num_labels']
        
        if not multi_label:
            assert num_labels >= 2, (
                "For single-label classification, num_labels should be >= 2"
            )

    def test_fp16_and_bf16_not_both_true(self, mimic_classifier_config):
        """Verify that fp16 and bf16 are not both enabled."""
        training_config = mimic_classifier_config['training']
        fp16 = training_config.get('fp16', False)
        bf16 = training_config.get('bf16', False)
        
        assert not (fp16 and bf16), (
            "Cannot enable both fp16 and bf16 simultaneously"
        )

    def test_eval_steps_less_than_total_steps(self, mimic_classifier_config):
        """Verify that eval_steps is reasonable."""
        training_config = mimic_classifier_config['training']
        eval_steps = training_config['eval_steps']
        
        # Should be positive
        assert eval_steps > 0, "eval_steps must be positive"
        
        # Should be reasonable (not too large)
        assert eval_steps < 10000, (
            "eval_steps seems very large, may never evaluate"
        )

    def test_batch_size_positive(self, mimic_classifier_config):
        """Verify that batch sizes are positive."""
        training_config = mimic_classifier_config['training']
        
        assert training_config['batch_size'] > 0
        assert training_config['eval_batch_size'] > 0

    def test_learning_rate_reasonable(self, mimic_classifier_config):
        """Verify that learning rate is in a reasonable range."""
        training_config = mimic_classifier_config['training']
        lr = training_config['learning_rate']
        
        # Convert to float if string
        if isinstance(lr, str):
            lr = float(lr)
        
        assert 1e-8 <= lr <= 1e-2, (
            f"Learning rate {lr} seems outside typical range [1e-8, 1e-2]"
        )


class TestConfigForCommonMistakes:
    """Test for common configuration mistakes that cause silent failures."""

    def test_task_not_missing_for_classification(self, mimic_classifier_config):
        """
        CRITICAL TEST: Ensure 'task' field is present in data config.
        
        This is the bug that caused the original issue:
        - Without 'task' field, dataset_factory defaults to 'pretrain'
        - In pretrain mode, ACES labels are never loaded
        - All examples get label=0, causing training to fail
        """
        data_config = mimic_classifier_config['data']
        
        assert 'task' in data_config, (
            "CRITICAL: 'task' field is missing from data config! "
            "Without this, the dataset will default to 'pretrain' mode and "
            "ACES labels will NOT be loaded, causing all labels to be 0."
        )

    def test_aces_label_dir_not_parent_directory(self, mimic_classifier_config):
        """
        Verify that aces_label_dir points to the specific task directory,
        not just the parent labels directory.
        
        Common mistake: setting aces_label_dir to '/path/to/labels' instead of
        '/path/to/labels/icu_mortality' or '/path/to/labels/readmission', etc.
        """
        data_config = mimic_classifier_config['data']
        aces_dir = data_config['aces_label_dir']
        
        # Check that train_labels.parquet exists in this directory
        train_labels_path = os.path.join(aces_dir, "train_labels.parquet")
        assert os.path.exists(train_labels_path), (
            f"aces_label_dir may be pointing to parent directory. "
            f"Expected to find train_labels.parquet at: {train_labels_path}\n"
            f"Current aces_label_dir: {aces_dir}\n"
            f"Did you mean to point to a subdirectory like 'icu_mortality'?"
        )

    def test_output_dir_is_absolute_path(self, mimic_classifier_config):
        """Verify that output_dir is an absolute path."""
        output_dir = mimic_classifier_config['training']['output_dir']
        assert os.path.isabs(output_dir), (
            f"output_dir should be an absolute path, got: {output_dir}"
        )

    def test_checkpoint_path_is_absolute(self, mimic_classifier_config):
        """Verify that pretrained_checkpoint is an absolute path."""
        checkpoint = mimic_classifier_config['model']['pretrained_checkpoint']
        assert os.path.isabs(checkpoint), (
            f"pretrained_checkpoint should be an absolute path, got: {checkpoint}"
        )


class TestDatasetSpecificConfigs:
    """Test that dataset-specific configs have the right parameters."""

    def test_mimic_config_has_mimic_specific_fields(self, mimic_classifier_config):
        """Verify MIMIC config has MIMIC-specific fields."""
        data_config = mimic_classifier_config['data']
        
        # MIMIC-specific fields
        mimic_fields = ['aces_label_dir', 'lookup_dir']
        for field in mimic_fields:
            assert field in data_config, (
                f"MIMIC config missing required field: {field}"
            )

    def test_mimic_config_does_not_have_cprd_fields(self, mimic_classifier_config):
        """Verify MIMIC config doesn't have CPRD-specific fields."""
        data_config = mimic_classifier_config['data']
        
        # CPRD-specific fields that shouldn't be in MIMIC config
        cprd_fields = [
            'labels_filepath',
            'medical_lookup_filepath',
            'lab_lookup_filepath',
            'region_lookup_filepath',
            'time_lookup_filepath',
            'cutoff_months',
        ]
        for field in cprd_fields:
            assert field not in data_config, (
                f"MIMIC config should not have CPRD-specific field: {field}"
            )


class TestConfigValidation:
    """Test that config values are valid."""

    def test_head_type_is_valid(self, mimic_classifier_config):
        """Verify that head_type is a supported value."""
        model_config = mimic_classifier_config['model']
        head_type = model_config.get('head_type', 'linear')
        
        valid_head_types = ['linear', 'mlp', 'deep_mlp']
        assert head_type in valid_head_types, (
            f"head_type must be one of {valid_head_types}, got: {head_type}"
        )

    def test_dataset_name_is_valid(self, mimic_classifier_config):
        """Verify that dataset name is supported."""
        dataset_name = mimic_classifier_config['data']['dataset']
        valid_datasets = ['MIMIC', 'CPRD']
        
        assert dataset_name in valid_datasets, (
            f"dataset must be one of {valid_datasets}, got: {dataset_name}"
        )

    def test_task_is_valid(self, mimic_classifier_config):
        """Verify that task is a valid value."""
        data_config = mimic_classifier_config['data']
        task = data_config.get('task', 'pretrain')
        
        valid_tasks = ['pretrain', 'prediction']
        assert task in valid_tasks, (
            f"task must be one of {valid_tasks}, got: {task}"
        )

    def test_label_column_specified_for_prediction(self, mimic_classifier_config):
        """Verify that label_column is specified when task='prediction'."""
        data_config = mimic_classifier_config['data']
        task = data_config.get('task', 'pretrain')
        
        if task == 'prediction':
            # label_column can be omitted (defaults to 'label'), but if present should be string
            label_column = data_config.get('label_column', 'label')
            assert isinstance(label_column, str), (
                "label_column must be a string"
            )
