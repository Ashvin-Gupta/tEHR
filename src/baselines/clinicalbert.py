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

    def _restore_weights_from_checkpoint(self, checkpoint_path: str):
        """Load the full sequence-classifier state dict (encoder + head)."""
        import os
        import torch

        print("\n" + "=" * 80)
        print("Loading weights from checkpoint (eval-only)...")
        print("=" * 80)
        print(f"  - Source checkpoint: {checkpoint_path}")

        weights_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: {checkpoint_path}"
            )

        try:
            from safetensors.torch import load_file as _st_load
            ckpt_state = _st_load(weights_file)
        except Exception:
            ckpt_state = torch.load(weights_file, map_location="cpu", weights_only=True)

        missing, unexpected = self.classifier_model.load_state_dict(
            ckpt_state, strict=False
        )
        print(f"  - Loaded {len(ckpt_state)} tensors into sequence classifier")
        if missing:
            print(f"  - Missing keys: {list(missing)[:8]}")
        if unexpected:
            print(f"  - Unexpected keys ignored: {list(unexpected)[:8]}")