"""
Baseline 6: XGBoost Classifier

Uses token type counts as features for XGBoost classification.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import xgboost as xgb

from src.baselines.base_baselines import BaseBaseline
from src.baselines.baseline_utils import setup_output_dir
from src.evaluations.evaluations import compute_baseline_metrics, plot_all_curves, save_results, print_results


def extract_token_features(patient_record: Dict) -> Dict[str, int]:
    """
    Extract token type counts as features from patient record.
    
    Args:
        patient_record: Patient record dictionary with 'tokens' key
    
    Returns:
        Dictionary of feature counts
    """
    tokens = patient_record.get('tokens', [])
    
    features = {
        'num_medical_tokens': 0,
        'num_lab_tokens': 0,
        'num_demographic_tokens': 0,
        'num_time_tokens': 0,
        'num_numeric_tokens': 0,
        'num_other_tokens': 0,
        'total_tokens': len(tokens)
    }
    
    for token in tokens:
        if isinstance(token, str):
            token_upper = token.upper()
            
            # Medical codes
            if token_upper.startswith('MEDICAL//'):
                features['num_medical_tokens'] += 1
            # Lab codes
            elif token_upper.startswith('LAB//'):
                features['num_lab_tokens'] += 1
            # Demographics
            elif token_upper.startswith('AGE') or token_upper.startswith('GENDER//') or token_upper.startswith('ETHNICITY//'):
                features['num_demographic_tokens'] += 1
            # Time intervals
            elif token_upper.startswith('<TIME_INTERVAL_'):
                features['num_time_tokens'] += 1
            # Numeric values
            elif token.replace('.', '', 1).replace('-', '', 1).isdigit():
                features['num_numeric_tokens'] += 1
            else:
                features['num_other_tokens'] += 1
        else:
            features['num_other_tokens'] += 1
    
    return features


class XGBoostBaseline(BaseBaseline):
    """
    XGBoost baseline using token type count features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.training_config = config.get("training", {})
        # Maintain backwards compatibility with configs that store output_dir under training.*
        self.output_dir = self.training_config.get("output_dir", self.output_dir)
        
        self.model = None
        self.feature_names = None
    
    def _load_binary_labels(self) -> Dict[int, int]:
        """
        Load binary labels (is_case) keyed by subject_id.

        XGBoost baselines operate on a binary task. We do not use the dataset's
        internal multi-class mapping.
        """
        labels_df = pd.read_csv(self.data_config["labels_filepath"])
        return pd.Series(labels_df["is_case"].values, index=labels_df["subject_id"]).to_dict()

    def extract_features_from_dataset(self, dataset, subject_to_is_case: Dict[int, int]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features and labels from dataset.
        
        For XGBoost, we access raw tokens directly from patient records to avoid
        translation overhead (we don't need translation, just token type counts).
        
        Args:
            dataset: UnifiedEHRDataset instance
        
        Returns:
            Tuple of (features_df, labels_array)
        """
        features_list = []
        labels_list = []
        
        print("Extracting features from dataset...")
        print("  - Accessing raw tokens directly (no translation needed for XGBoost)")
        
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

            # Apply the dataset's cutoff logic (handles both cases and controls)
            if dataset.cutoff_months is not None:
                token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, int(label))

            # Convert token IDs to strings
            token_strings = [dataset.id_to_token.get(tid, "") for tid in token_ids]
            
            # Filter out tokens containing "cancer" (case-insensitive)
            token_strings = [token for token in token_strings if 'cancer' not in str(token).lower()]
            
            # Create sample dict with tokens
            sample = {'tokens': token_strings}
            
            # Extract features
            features = extract_token_features(sample)
            features_list.append(features)
            
            # Use is_case directly (already binary: 0 = Control, 1 = Pancreatic Cancer)
            labels_list.append(int(label))
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        labels_array = np.array(labels_list)
        
        # Verify binary labels
        unique_labels = np.unique(labels_array)
        print(f"  - Unique labels from is_case: {unique_labels}")
        if len(unique_labels) > 2 or not all(l in [0, 1] for l in unique_labels):
            raise ValueError(f"Expected binary labels [0, 1] from is_case, got {unique_labels}")
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        return features_df, labels_array
    
    def load_model(self):
        """Create XGBoost classifier (no neural model loading)."""
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": self.model_config.get("max_depth", 6),
            "learning_rate": self.model_config.get("learning_rate", 0.1),
            "n_estimators": self.model_config.get("n_estimators", 100),
            "subsample": self.model_config.get("subsample", 0.8),
            "colsample_bytree": self.model_config.get("colsample_bytree", 0.8),
            "min_child_weight": self.model_config.get("min_child_weight", 1),
            "gamma": self.model_config.get("gamma", 0),
            "reg_alpha": self.model_config.get("reg_alpha", 0),
            "reg_lambda": self.model_config.get("reg_lambda", 1),
            "random_state": 42,
            "n_jobs": self.training_config.get("n_jobs", -1),
            "early_stopping_rounds": self.training_config.get("early_stopping_rounds", 10),
        }
        self.model = xgb.XGBClassifier(**params)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        print("\n" + "=" * 80)
        print("Training XGBoost model...")
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

    def predict(self, dataset) -> np.ndarray:
        """
        Predict probabilities for the positive class on a dataset.

        Note: XGBoost baselines use binary `is_case` labels from the labels file,
        not the dataset's multi-class label mapping. This method exists primarily
        to satisfy the `BaseBaseline` interface; evaluation in this baseline is
        driven by the explicit (X, y) extraction in `run()`.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load_model() and train() first.")

        subject_to_is_case = self._load_binary_labels()
        X, _y = self.extract_features_from_dataset(dataset, subject_to_is_case)
        probs = self.model.predict_proba(X)[:, 1]
        return probs.astype(np.float32)
    
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
        
        # Feature importance (convert numpy types to native Python types)
        feature_importance = {k: float(v) for k, v in zip(self.feature_names, self.model.feature_importances_)}
        feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Most Important Features:")
        print("-" * 50)
        for feature, importance in feature_importance_sorted[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        results['feature_importance'] = feature_importance
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        return metrics
    
    def save_model(self):
        """Save the trained model."""
        model_path = os.path.join(self.output_dir, 'models', 'xgboost_model.json')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        print(f"\n  - Model saved to: {model_path}")
    
    def run(self):
        """Run the complete baseline training and evaluation pipeline."""
        setup_output_dir(self.output_dir)

        # Ensure we load token-form datasets for XGBoost.
        self.data_config["format"] = "tokens"
        datasets = self.load_datasets(splits=["train", "tuning", "held_out"])
        subject_to_is_case = self._load_binary_labels()
        
        # Extract features
        print("\n" + "=" * 80)
        print("Extracting features from datasets...")
        print("=" * 80)
        
        X_train, y_train = self.extract_features_from_dataset(datasets["train"], subject_to_is_case)
        X_val, y_val = self.extract_features_from_dataset(datasets["tuning"], subject_to_is_case)
        X_test, y_test = self.extract_features_from_dataset(datasets["held_out"], subject_to_is_case)
        
        print(f"\nFeature extraction complete:")
        print(f"  - Train: {len(X_train)} samples, {len(X_train.columns)} features")
        print(f"  - Validation: {len(X_val)} samples")
        print(f"  - Test: {len(X_test)} samples")
        print(f"\nFeature names: {list(X_train.columns)}")
        
        # Create model
        self.load_model()
        
        # Train
        self.train(X_train, y_train, X_val, y_val)
        
        # Save model
        self.save_model()
        
        # Evaluate
        val_metrics = self.evaluate(X_val, y_val, "tuning")
        test_metrics = self.evaluate(X_test, y_test, "held_out")
        
        # Save summary (convert numpy types to native Python types for JSON serialization)
        summary = {
            'validation': {k: float(v) for k, v in val_metrics.items()},
            'test': {k: float(v) for k, v in test_metrics.items()},
            'feature_names': self.feature_names
        }
        save_results(summary, self.output_dir, 'summary.json')
        
        print("\n" + "=" * 80)
        print("XGBoost Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

