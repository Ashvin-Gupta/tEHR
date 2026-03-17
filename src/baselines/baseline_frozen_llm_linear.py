"""
Baseline 2: Frozen LLM + Linear Classifier

Uses base Qwen model (no pretraining) with frozen parameters, trains only linear head.
"""
from unsloth import FastLanguageModel
from src.baselines.base_llm_classifier import BaseLLMClassifierBaseline

class FrozenLLMLinearBaseline(BaseLLMClassifierBaseline):
    """Base LLM with frozen parameters, trains only the linear classification head."""

    def load_model(self):
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["model_name"],
            max_seq_length=self.model_config.get("max_length", 8192),
            dtype=None,
            load_in_4bit=self.model_config.get("load_in_4bit", True),
            device_map="auto",
        )