"""
Baseline: LLM with LoRA Adapters + Linear Classifier.
Set ``model.use_unsloth: false`` in config for the HF+BitsAndBytes path.
"""
from src.baselines.base_llm_classifier import BaseLLMClassifierBaseline
from unsloth import FastLanguageModel

class LoRALLMLinearBaseline(BaseLLMClassifierBaseline):
    """Base LLM with LoRA adapters, trains LoRA + linear classification head."""

    def load_model(self):
        use_unsloth = self.model_config.get("use_unsloth", True)
        lora_cfg = self.config.get("lora", {})

        if use_unsloth:
            
            self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config["model_name"],
                max_seq_length=self.model_config.get("max_length", 8192),
                dtype=None,
                load_in_4bit=self.model_config.get("load_in_4bit", True),
                device_map="auto",
            )
            self.base_model = FastLanguageModel.get_peft_model(
                self.base_model,
                r=lora_cfg.get("r", 16),
                target_modules=lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                lora_alpha=lora_cfg.get("lora_alpha", 16),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                bias=lora_cfg.get("bias", "none"),
                use_gradient_checkpointing=self.training_config.get("gradient_checkpointing", "unsloth"),
                random_state=42,
                use_rslora=lora_cfg.get("use_rslora", True),
            )
        else:
            from src.training.model_loader import load_hf_model, apply_lora_peft
            self.base_model, self.tokenizer = load_hf_model(
                model_name=self.model_config["model_name"],
                max_seq_length=self.model_config.get("max_length", 8192),
                load_in_4bit=self.model_config.get("load_in_4bit", True),
                local_rank=getattr(self, "local_rank", 0),
            )
            self.base_model = apply_lora_peft(self.base_model, lora_cfg)

    def _get_trainable_keywords(self):
        return ["lora_"]