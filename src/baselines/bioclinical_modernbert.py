"""
Experiment: bioclinical_modernbert — BioClinical-ModernBERT-base sequence classifier.

Same training path as ``clinicalbert`` (HF sequence classification on EHR text),
with ModernBERT's 8192-token context window.
https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.baselines.clinicalbert import ClinicalBERTBaseline


class BioClinicalModernBERTBaseline(ClinicalBERTBaseline):
    """Fine-tunes BioClinical-ModernBERT-base for EHR classification."""

    def load_model(self):
        model_name = self.model_config.get(
            "model_name", "thomas-sounack/BioClinical-ModernBERT-base"
        )
        num_labels = self.model_config.get("num_labels", 2)
        attn = self.model_config.get("attn_implementation", "flash_attention_2")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.model_max_length = int(self.model_config.get("max_length", 8192))

        load_kwargs = {"num_labels": num_labels}
        if attn:
            load_kwargs["attn_implementation"] = attn

        try:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )
        except Exception as exc:
            if "attn_implementation" in load_kwargs:
                print(
                    f"  - attn_implementation={attn!r} failed ({type(exc).__name__}: {exc}); "
                    "retrying with the default attention backend"
                )
                load_kwargs.pop("attn_implementation")
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **load_kwargs
                )
            else:
                raise

        # ModernBERT torch.compile path is brittle under HF Trainer.
        if hasattr(self.base_model.config, "reference_compile"):
            self.base_model.config.reference_compile = False
        if hasattr(self.base_model.config, "use_cache"):
            self.base_model.config.use_cache = False
        print(
            f"  - Loaded {model_name}  "
            f"max_length={self.model_config.get('max_length', 8192)}"
        )
