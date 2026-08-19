"""
Experiment: clinicalbert — Bio_ClinicalBERT sequence classifier.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.baselines.hf_classifier import BaseLLMClassifierBaseline


class ClinicalBERTBaseline(BaseLLMClassifierBaseline):
    """Fine-tunes ClinicalBERT for EHR classification."""

    def load_model(self):
        model_name = self.model_config.get("model_name", "emilyalsentzer/Bio_ClinicalBERT")
        num_labels = self.model_config.get("num_labels", 2)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store as base_model so create_classifier() can reference it
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def create_classifier(self):
        # BERT already has a classification head — skip the LLMClassifier wrapper
        self.classifier_model = self.base_model