"""
Baseline: LLM with LoRA Adapters + Linear Classifier
"""
from unsloth import FastLanguageModel
from src.baselines.base_llm_classifier import BaseLLMClassifierBaseline


class LoRALLMLinearBaseline(BaseLLMClassifierBaseline):
    """Base LLM with LoRA adapters, trains LoRA + linear classification head."""

    def load_model(self):
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config["model_name"],
            max_seq_length=self.model_config.get("max_length", 8192),
            dtype=None,
            load_in_4bit=self.model_config.get("load_in_4bit", True),
            device_map="auto",
        )
        lora_cfg = self.config.get("lora", {})
        self.base_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=lora_cfg.get("r", 16),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=lora_cfg.get("lora_alpha", 16),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            use_gradient_checkpointing=self.training_config.get("gradient_checkpointing", "unsloth"),
            random_state=42,
            use_rslora=lora_cfg.get("use_rslora", True),
        )

    def _get_trainable_keywords(self):
        return ["lora_"]