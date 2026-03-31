"""
Baseline: Logistic regression on simple demographics/event count features.

Features:
  - age
  - gender (binary encoded)
  - number_of_events
"""
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.base_baselines import BaseBaseline
from src.baselines.baseline_utils import setup_output_dir
from src.evaluations.evaluations import compute_baseline_metrics, plot_all_curves, save_results, print_results


class LogRegDemoEventsBaseline(BaseBaseline):
    """
    Logistic regression baseline using only:
      - age
      - gender
      - number_of_events
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.training_config = config.get("training", {})
        self.debug = 0
        # Backwards compatibility with configs that place output_dir in training.*
        self.output_dir = self.training_config.get("output_dir", self.output_dir)
        self.model = None

    def _load_binary_labels(self) -> Dict[int, int]:
        labels_df = pd.read_csv(self.data_config["labels_filepath"])
        return pd.Series(labels_df["is_case"].values, index=labels_df["subject_id"]).to_dict()

    @staticmethod
    def _extract_age(token_strings):
        age = None
        for i, token in enumerate(token_strings):
            if not isinstance(token, str):
                continue
            if token == "AGE" or token.startswith("AGE:") or token.startswith("AGE "):
                if ":" in token:
                    age_str = token.split(":")[-1].strip()
                    try:
                        age = float(age_str)
                    except ValueError:
                        pass
                elif token.startswith("AGE "):
                    age_str = token.replace("AGE", "").strip()
                    try:
                        age = float(age_str)
                    except ValueError:
                        pass
                elif i + 1 < len(token_strings):
                    next_token = str(token_strings[i + 1])
                    if next_token.replace(".", "", 1).replace("-", "", 1).isdigit():
                        age = float(next_token)
                if age is not None:
                    break
        return age

    @staticmethod
    def _extract_gender(token_strings):
        for token in token_strings:
            if isinstance(token, str) and token.startswith("GENDER//"):
                return token.split("//")[-1].strip().lower()
        return None

    @staticmethod
    def _gender_to_float(gender):
        if gender is None:
            return np.nan
        if gender == "male":
            return 1.0
        if gender == "female":
            return 0.0
        return np.nan

    @staticmethod
    def _count_events(token_strings):
        return float(
            sum(
                1
                for t in token_strings
                if isinstance(t, str)
                and (
                    t.startswith("MEDICAL//")
                    or t.startswith("LAB//")
                    or t.startswith("MEASUREMENT//")
                )
            )
        )

    def extract_features_from_dataset(
        self, dataset, subject_to_is_case: Dict[int, int]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        feature_rows = []
        labels = []

        for patient_record in dataset.patient_records:
            subject_id = patient_record["subject_id"]
            label = subject_to_is_case.get(subject_id)
            if pd.isna(label) if hasattr(pd, "isna") else (label is None):
                continue

            token_ids = patient_record["tokens"]
            timestamps = patient_record["timestamps"]
            if dataset.cutoff_months is not None:
                token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, int(label))

            token_strings = [dataset.id_to_token.get(tid, "") for tid in token_ids]

            age = self._extract_age(token_strings)
            gender = self._extract_gender(token_strings)
            number_of_events = self._count_events(token_strings)

            feature_rows.append(
                {
                    "age": age,
                    "gender": self._gender_to_float(gender),
                    "number_of_events": number_of_events,
                }
            )
            labels.append(int(label))

        features_df = pd.DataFrame(feature_rows, columns=["age", "gender", "number_of_events"])
        # Print the entire feature matrix in one batch for observation
        print("\n=== Feature Matrix (X) being fed into Logistic Regression ===")
        print(features_df)
        print("=== End Feature Matrix ===\n")
        return features_df, np.array(labels, dtype=np.int64)

    def load_model(self):
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.model_config.get("max_iter", 1000),
                        C=self.model_config.get("C", 1.0),
                        class_weight=self.model_config.get("class_weight", "balanced"),
                        solver=self.model_config.get("solver", "lbfgs"),
                        random_state=self.model_config.get("random_state", 42),
                    ),
                ),
            ]
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def predict(self, dataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load_model() and train() first.")
        subject_to_is_case = self._load_binary_labels()
        X, _ = self.extract_features_from_dataset(dataset, subject_to_is_case)
        probs = self.predict_proba(X)
        return probs.astype(np.float32)

    def evaluate(self, X, y, split_name: str = "validation") -> Dict[str, float]:
        probs = self.predict_proba(X)
        metrics = compute_baseline_metrics(y, probs)
        print_results(metrics, split_name)

        plot_dir = os.path.join(self.output_dir, "plots", split_name)
        plot_all_curves(y, probs, plot_dir)

        results = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "labels": [int(label) for label in y.tolist()],
            "probs": [float(prob) for prob in probs.tolist()],
        }
        save_results(results, os.path.join(self.output_dir, "results"), f"{split_name}_results.json")
        return metrics

    def save_model(self):
        # Persist lightweight model parameters and coefficients for reproducibility.
        clf = self.model.named_steps["clf"]
        scaler = self.model.named_steps["scaler"]
        model_payload = {
            "feature_names": ["age", "gender", "number_of_events"],
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": clf.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "model_config": self.model_config,
        }
        save_results(model_payload, os.path.join(self.output_dir, "models"), "logreg_model.json")

    def run(self):
        setup_output_dir(self.output_dir)
        self.data_config["format"] = "tokens"
        datasets = self.load_datasets(splits=["train", "tuning", "held_out"])
        subject_to_is_case = self._load_binary_labels()

        X_train, y_train = self.extract_features_from_dataset(datasets["train"], subject_to_is_case)
        X_val, y_val = self.extract_features_from_dataset(datasets["tuning"], subject_to_is_case)
        X_test, y_test = self.extract_features_from_dataset(datasets["held_out"], subject_to_is_case)

        self.load_model()
        self.train(X_train, y_train)
        self.save_model()

        val_metrics = self.evaluate(X_val, y_val, "tuning")
        test_metrics = self.evaluate(X_test, y_test, "held_out")

        summary = {
            "validation": {k: float(v) for k, v in val_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
            "feature_names": ["age", "gender", "number_of_events"],
        }
        save_results(summary, self.output_dir, "summary.json")
        return val_metrics, test_metrics
