"""
Baseline 6b: XGBoost Classifier with Bag-of-Words Features

Uses each unique token as a feature (bag-of-words approach) instead of aggregated counts.

This implementation filters tokens to include only:
- Medical events (MEDICAL//...)
- Lab test names (LAB//...) - without their numeric values
- Measurement codes (MEASUREMENT//...)
- Demographics (AGE, GENDER//..., ETHNICITY//..., REGION//...)
- Lifestyle codes (LIFESTYLE//...)
- Special tokens (<start>, <end>, <unknown>, MEDS_BIRTH)

Excludes:
- Time intervals (<time_interval_...>)
- Numeric values
- Units (e.g., "mmol/L", "kg", "%")
- Lab value categories (low, normal, high, etc.)
- Quantile values (Q1, Q2, Q3, Q4)
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from tqdm import tqdm
import xgboost as xgb
from collections import Counter

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results


def should_include_token(token_str: str) -> bool:
    """
    Check if a token should be included in the bag-of-words features.
    
    Includes:
    - Medical events (MEDICAL//...)
    - Lab test names (LAB//...) - but not their values
    - Measurement codes (MEASUREMENT//...)
    - Demographics (AGE, GENDER//..., ETHNICITY//..., REGION//...)
    - Lifestyle (LIFESTYLE//...)
    - Special tokens (<start>, <end>, <unknown>, MEDS_BIRTH)
    
    Excludes:
    - Time intervals (<time_interval_...>)
    - Numeric values (pure numbers)
    - Units (short strings that don't match code patterns)
    - Lab value categories (low, normal, high, very low, very high)
    - Quantile values (Q1, Q2, Q3, Q4)
    
    Args:
        token_str: Token string to check
    
    Returns:
        True if token should be included, False otherwise
    """
    if not isinstance(token_str, str) or not token_str:
        return False
    
    token_upper = token_str.upper()
    token_lower = token_str.lower()
    
    # Exclude time intervals
    if token_str.startswith('<time_interval_'):
        return False
    
    # Exclude numeric values (pure numbers, including negative and decimal)
    if token_str.replace('.', '', 1).replace('-', '', 1).isdigit():
        return False
    
    # Exclude quantile values (Q1, Q2, Q3, Q4)
    if token_str.startswith('Q') and len(token_str) <= 4 and token_str[1:].isdigit():
        return False
    
    # Exclude lab value categories
    if token_lower in ['low', 'normal', 'high', 'very low', 'very high']:
        return False
    
    # Include medical codes
    if token_upper.startswith('MEDICAL//'):
        return True
    
    # Include lab test names (but not their values - values are numeric and already excluded above)
    if token_upper.startswith('LAB//'):
        return True
    
    # Include measurement codes
    if token_upper.startswith('MEASUREMENT//'):
        return True
    
    # Include demographics
    if token_upper.startswith(('AGE:', 'AGE', 'GENDER//', 'ETHNICITY//', 'REGION//')):
        return True
    
    # Include lifestyle
    if token_upper.startswith('LIFESTYLE//'):
        return True
    
    # Include special tokens
    if token_str in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
        return True
    
    # Exclude units - tokens that are short strings with unit-like characters
    # but don't match any known code pattern
    # Common units: "mmol/L", "mg/dL", "kg", "cm", "%", etc.
    if len(token_str) <= 20:
        # Check if it contains unit-like characters but isn't a code
        has_unit_chars = any(char in token_str for char in ['/', '%', '°', 'µ', 'μ'])
        # Or if it's a very short string (1-5 chars) that's all lowercase/uppercase letters
        is_short_alpha = len(token_str) <= 5 and token_str.replace(' ', '').isalpha()
        
        # If it doesn't match any code pattern and looks like a unit, exclude it
        if (has_unit_chars or is_short_alpha) and not token_upper.startswith((
            '<', 'MEDICAL', 'LAB', 'MEASUREMENT', 'AGE', 'GENDER', 
            'ETHNICITY', 'REGION', 'LIFESTYLE'
        )):
            return False
    
    # By default, exclude unknown tokens to be conservative
    # Only include tokens that explicitly match known patterns
    return False


def extract_bow_features(patient_record: Dict, token_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Extract bag-of-words features from patient record.
    
    Creates a feature vector where each position corresponds to a unique token,
    and the value is the count of that token in the patient's record.
    Only includes tokens that pass the should_include_token filter.
    
    Args:
        patient_record: Patient record dictionary with 'tokens' key
        token_to_idx: Mapping from token string to feature index
    
    Returns:
        Feature vector as numpy array
    """
    tokens = patient_record.get('tokens', [])
    
    # Filter tokens and count occurrences
    filtered_tokens = [str(token) for token in tokens if should_include_token(str(token))]
    token_counts = Counter(filtered_tokens)
    
    # Create feature vector
    feature_vector = np.zeros(len(token_to_idx), dtype=np.float32)
    for token, count in token_counts.items():
        if token in token_to_idx:
            feature_vector[token_to_idx[token]] = count
    
    return feature_vector


class XGBoostBOWBaseline:
    """
    XGBoost baseline using bag-of-words token features.
    
    Only includes medical events, lab test names (without values), demographics,
    and lifestyle tokens. Excludes time, units, numbers, and lab values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the XGBoost BOW baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/xgboost_bow')
        
        self.model = None
        self.token_to_idx = None
        self.idx_to_token = None
        self.vocab_size = None
    
    def build_vocabulary(self, datasets: Dict[str, Any]) -> Dict[str, int]:
        """
        Build vocabulary from all datasets.
        
        Args:
            datasets: Dictionary of datasets
        
        Returns:
            Mapping from token string to feature index
        """
        print("\n" + "=" * 80)
        print("Building vocabulary from datasets...")
        print("=" * 80)
        
        all_tokens = set()
        
        # Collect all unique tokens (filtered to exclude time, units, numbers, lab values)
        for split_name, dataset in datasets.items():
            print(f"  - Processing {split_name} split...")
            for i in tqdm(range(len(dataset.patient_records)), desc=f"Collecting tokens from {split_name}"):
                patient_record = dataset.patient_records[i]
                token_ids = patient_record.get('tokens', [])
                
                # Convert token IDs to strings and filter
                for token_id in token_ids:
                    token_str = str(dataset.id_to_token_map.get(token_id, ""))
                    # Filter out tokens containing "cancer" and apply other filters
                    if token_str and 'cancer' not in token_str.lower() and should_include_token(token_str):
                        all_tokens.add(token_str)
        
        # Create vocabulary mapping
        sorted_tokens = sorted(list(all_tokens))
        self.vocab_size = len(sorted_tokens)
        
        self.token_to_idx = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        print(f"  - Vocabulary size: {self.vocab_size:,} unique tokens (filtered: medical events, lab test names, demographics only)")
        
        # Print some statistics about what was included
        included_types = {
            'medical': sum(1 for t in sorted_tokens if t.upper().startswith('MEDICAL//')),
            'lab': sum(1 for t in sorted_tokens if t.upper().startswith('LAB//')),
            'measurement': sum(1 for t in sorted_tokens if t.upper().startswith('MEASUREMENT//')),
            'demographic': sum(1 for t in sorted_tokens if t.upper().startswith(('AGE', 'GENDER//', 'ETHNICITY//', 'REGION//'))),
            'lifestyle': sum(1 for t in sorted_tokens if t.upper().startswith('LIFESTYLE//')),
            'special': sum(1 for t in sorted_tokens if t in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH'])
        }
        print(f"  - Token breakdown:")
        for token_type, count in included_types.items():
            if count > 0:
                print(f"    {token_type}: {count:,}")
        
        return self.token_to_idx
    
    def extract_features_from_dataset(self, dataset, token_to_idx: Dict[str, int]) -> tuple:
        """
        Extract bag-of-words features and labels from dataset.
        
        Args:
            dataset: UnifiedEHRDataset instance
            token_to_idx: Mapping from token string to feature index
        
        Returns:
            Tuple of (features_array, labels_array)
        """
        features_list = []
        labels_list = []
        
        print("Extracting bag-of-words features from dataset...")
        
        # Load labels file directly to get binary is_case labels
        labels_df = pd.read_csv(self.data_config['labels_filepath'])
        subject_to_is_case = pd.Series(
            labels_df['is_case'].values,
            index=labels_df['subject_id']
        ).to_dict()
        
        # Access patient records directly to get raw tokens without translation
        for i in tqdm(range(len(dataset.patient_records)), desc="Processing samples"):
            patient_record = dataset.patient_records[i]
            subject_id = patient_record['subject_id']
            
            # Get binary label (is_case: 0 = Control, 1 = Pancreatic Cancer)
            label = subject_to_is_case.get(subject_id)
            if pd.isna(label) if hasattr(pd, 'isna') else (label is None):
                continue  # Skip patients without labels
            
            # Get raw token IDs and convert to strings using vocab
            token_ids = patient_record['tokens']
            timestamps = patient_record['timestamps']
            
            # Apply time cutoff if needed (same logic as dataset)
            if dataset.cutoff_months is not None:
                actual_cutoff = dataset.cutoff_months
                cancer_date = dataset.subject_to_cancer_date.get(subject_id)
                
                if pd.notna(cancer_date) if hasattr(pd, 'notna') else (cancer_date is not None):
                    cutoff_date = cancer_date - pd.DateOffset(months=actual_cutoff)
                    cutoff_timestamp = cutoff_date.timestamp()
                    
                    truncated_ids = []
                    for j, ts in enumerate(timestamps):
                        token_str = dataset.id_to_token_map.get(token_ids[j], "")
                        is_end_token = (token_str == '<end>')
                        if ts == 0 or (ts is not None and ts < cutoff_timestamp) or is_end_token:
                            truncated_ids.append(token_ids[j])
                    token_ids = truncated_ids
            
            # Convert token IDs to strings
            token_strings = [dataset.id_to_token_map.get(tid, "") for tid in token_ids]
            
            # Filter out tokens containing "cancer" (case-insensitive)
            token_strings = [token for token in token_strings if 'cancer' not in str(token).lower()]
            
            # Create sample dict with tokens
            sample = {'tokens': token_strings}
            
            # Extract bag-of-words features
            features = extract_bow_features(sample, token_to_idx)
            features_list.append(features)
            
            # Use is_case directly (already binary: 0 = Control, 1 = Pancreatic Cancer)
            labels_list.append(int(label))
        
        # Convert to numpy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        # Verify binary labels
        unique_labels = np.unique(labels_array)
        print(f"  - Unique labels from is_case: {unique_labels}")
        print(f"  - Feature matrix shape: {features_array.shape}")
        print(f"  - Sparsity: {(features_array == 0).sum() / features_array.size * 100:.2f}% zeros")
        
        if len(unique_labels) > 2 or not all(l in [0, 1] for l in unique_labels):
            raise ValueError(f"Expected binary labels [0, 1] from is_case, got {unique_labels}")
        
        return features_array, labels_array
    
    def create_model(self):
        """Create XGBoost classifier."""
        print("\n" + "=" * 80)
        print("Creating XGBoost classifier (Bag-of-Words)...")
        print("=" * 80)
        
        # Get hyperparameters from config
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': self.model_config.get('max_depth', 6),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'n_estimators': self.model_config.get('n_estimators', 100),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'min_child_weight': self.model_config.get('min_child_weight', 1),
            'gamma': self.model_config.get('gamma', 0),
            'reg_alpha': self.model_config.get('reg_alpha', 0),
            'reg_lambda': self.model_config.get('reg_lambda', 1),
            'random_state': 42,
            'n_jobs': self.training_config.get('n_jobs', -1),
            'early_stopping_rounds': self.training_config.get('early_stopping_rounds', 10),
            'tree_method': 'hist'  # Use histogram method for faster training with many features
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        print(f"  - XGBoost model created with parameters:")
        for key, value in params.items():
            if key not in ['n_jobs', 'early_stopping_rounds']:
                print(f"    {key}: {value}")
        print(f"    early_stopping_rounds: {params['early_stopping_rounds']}")
        print(f"    Vocabulary size: {self.vocab_size:,} features")
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        print("\n" + "=" * 80)
        print("Training XGBoost model (Bag-of-Words)...")
        print("=" * 80)
        
        # Train with early stopping
        # Note: early_stopping_rounds is set during model initialization in newer XGBoost versions
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.training_config.get('verbose', True)
        )
        
        print("\nTraining completed!")
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of probabilities for positive class
        """
        probs = self.model.predict_proba(X)
        return probs[:, 1]  # Return probability of positive class
    
    def evaluate(self, X, y, split_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Feature matrix
            y: True labels
            split_name: Name of the split
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n" + "=" * 80)
        print(f"Evaluating on {split_name} set...")
        print("=" * 80)
        
        # Get predictions
        probs = self.predict_proba(X)
        
        # Compute metrics
        metrics = compute_baseline_metrics(y, probs)
        
        # Print results
        print_results(metrics, split_name)
        
        # Save all plots (ROC, PR, and Calibration)
        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(y, probs, plot_dir)
        
        # Save predictions (convert numpy types to native Python types for JSON serialization)
        results = {
            'metrics': {k: float(v) for k, v in metrics.items()},
            'labels': [int(label) for label in y.tolist()],
            'probs': [float(prob) for prob in probs.tolist()]
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        # Feature importance (top features)
        feature_importance = {self.idx_to_token[idx]: float(imp) 
                             for idx, imp in enumerate(self.model.feature_importances_)
                             if imp > 0}
        feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 20 Most Important Token Features:")
        print("-" * 50)
        for feature, importance in feature_importance_sorted[:20]:
            print(f"  {feature}: {importance:.4f}")
        
        results['feature_importance'] = {k: float(v) for k, v in feature_importance.items()}
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        return metrics
    
    def save_model(self):
        """Save the trained model."""
        model_path = os.path.join(self.output_dir, 'models', 'xgboost_bow_model.json')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        
        # Also save vocabulary mapping
        vocab_path = os.path.join(self.output_dir, 'models', 'vocabulary.json')
        import json
        with open(vocab_path, 'w') as f:
            json.dump({
                'token_to_idx': self.token_to_idx,
                'vocab_size': self.vocab_size
            }, f, indent=2)
        
        print(f"\n  - Model saved to: {model_path}")
        print(f"  - Vocabulary saved to: {vocab_path}")
    
    def run(self):
        """Run the complete baseline training and evaluation pipeline."""
        # Setup
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='tokens'
        )
        
        # Build vocabulary from all splits
        self.build_vocabulary(datasets)
        
        # Extract features
        print("\n" + "=" * 80)
        print("Extracting bag-of-words features from datasets...")
        print("=" * 80)
        
        X_train, y_train = self.extract_features_from_dataset(datasets['train'], self.token_to_idx)
        X_val, y_val = self.extract_features_from_dataset(datasets['tuning'], self.token_to_idx)
        X_test, y_test = self.extract_features_from_dataset(datasets['held_out'], self.token_to_idx)
        
        print(f"\nFeature extraction complete:")
        print(f"  - Train: {len(X_train)} samples, {X_train.shape[1]:,} features")
        print(f"  - Validation: {len(X_val)} samples")
        print(f"  - Test: {len(X_test)} samples")
        
        # Create model
        self.create_model()
        
        # Train
        self.train(X_train, y_train, X_val, y_val)
        
        # Save model
        self.save_model()
        
        # Evaluate
        val_metrics = self.evaluate(X_val, y_val, 'validation')
        test_metrics = self.evaluate(X_test, y_test, 'test')
        
        # Save summary
        summary = {
            'validation': {k: float(v) for k, v in val_metrics.items()},
            'test': {k: float(v) for k, v in test_metrics.items()},
            'vocab_size': self.vocab_size
        }
        save_results(summary, self.output_dir, 'summary.json')
        
        print("\n" + "=" * 80)
        print("XGBoost Bag-of-Words Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

