"""
Experiment: ablation_nopretrain_frozen — frozen base LLM, train linear head only.

Set ``model.use_unsloth: false`` for HuggingFace + BitsAndBytes instead of Unsloth.
"""
from src.baselines.hf_classifier import BaseLLMClassifierBaseline
from unsloth import FastLanguageModel

class AblationNopretrainFrozenBaseline(BaseLLMClassifierBaseline):
    """Base LLM with frozen parameters, trains only the linear classification head."""

    def load_model(self):
        use_unsloth = self.model_config.get("use_unsloth", True)
        if use_unsloth:
            
            # Keep the whole 4-bit model on one device. device_map="auto" plus
            # HF Trainer DataParallel (multiple visible GPUs) causes
            # cuda:0 vs cuda:1 mismatches in bitsandbytes matmul.
            self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config["model_name"],
                max_seq_length=self.model_config.get("max_length", 8192),
                dtype=None,
                load_in_4bit=self.model_config.get("load_in_4bit", True),
                device_map={"": 0},
            )
        else:
            from src.training.model_loader import load_hf_model
            self.base_model, self.tokenizer = load_hf_model(
                model_name=self.model_config["model_name"],
                max_seq_length=self.model_config.get("max_length", 8192),
                load_in_4bit=self.model_config.get("load_in_4bit", True),
                local_rank=getattr(self, "local_rank", 0),
            )
