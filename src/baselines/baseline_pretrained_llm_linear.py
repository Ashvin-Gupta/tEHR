"""
Baseline: Pretrained LLM + Linear Classifier

Loads from a saved pretraining checkpoint (LLM + LoRA adapters), freezes all 
parameters, and trains only the linear classification head.
"""
from src.baselines.base_llm_classifier import BaseLLMClassifierBaseline
from src.training.model_loader import load_pretrained_lora_model


class PretrainedLLMLinearBaseline(BaseLLMClassifierBaseline):
    """Pretrained LLM (with LoRA) with frozen parameters and trainable linear head."""

    def load_model(self):
        pretrained_checkpoint = self.model_config.get("pretrained_checkpoint")
        if not pretrained_checkpoint:
            raise ValueError("'model.pretrained_checkpoint' must be set in config")
        self.base_model, self.tokenizer = load_pretrained_lora_model(
            pretrained_checkpoint=pretrained_checkpoint,
            max_seq_length=self.model_config.get("max_length", 8192),
            load_in_4bit=self.model_config.get("load_in_4bit", True),
            local_rank=self.local_rank,
        )