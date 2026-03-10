"""
High-level interface for generating future trajectories from EHR records
using a pretrained LLM with LoRA adapters.
"""
from typing import Dict, Any, Optional

from unsloth import FastLanguageModel

from src.training.model_loader import load_pretrained_lora_model
from src.inference.generator import run_ehr_inference


DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert clinician and medical summarisation model.\n"
    "Given this patient's longitudinal EHR record, continue the trajectory "
    "by describing likely future events, tests, and outcomes in a coherent narrative.\n\n"
    "Patient EHR record:\n"
    "{ehr_text}\n\n"
    "Continue the trajectory:"
)


class EHRTrajectoryGenerator:
    """
    Wrapper around a pretrained LLM + LoRA adapters for trajectory generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary with keys:
                - model:
                    - checkpoint_dir: path to pretrained LoRA checkpoint
                    - max_length: max context length
                    - load_in_4bit: bool
                - generation (optional):
                    - max_new_tokens
                    - temperature
                    - top_p
                    - top_k
                    - repetition_penalty
                - prompt (optional):
                    - template: custom prompt template with "{ehr_text}"
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.gen_config = config.get("generation", {})
        self.prompt_config = config.get("prompt", {})

        checkpoint_dir = self.model_config.get("checkpoint_dir")
        if checkpoint_dir is None:
            raise ValueError("`model.checkpoint_dir` must be provided in the config.")

        max_seq_length = int(self.model_config.get("max_length", 8192))
        load_in_4bit = bool(self.model_config.get("load_in_4bit", True))
        local_rank = int(self.model_config.get("local_rank", 0))

        self.model, self.tokenizer = load_pretrained_lora_model(
            pretrained_checkpoint=checkpoint_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            local_rank=local_rank,
        )

    def _build_prompt(self, ehr_text: str, extra_instruction: Optional[str] = None) -> str:
        template = self.prompt_config.get("template", DEFAULT_PROMPT_TEMPLATE)
        if extra_instruction:
            template = template + "\n\nAdditional instruction:\n" + extra_instruction
        return template.format(ehr_text=ehr_text)

    def generate(
        self,
        ehr_text: str,
        extra_instruction: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """
        Generate a future trajectory continuation for a single EHR record.
        """
        prompt = self._build_prompt(ehr_text, extra_instruction=extra_instruction)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.gen_config.get("max_new_tokens", 256),
            "temperature": temperature or self.gen_config.get("temperature", 0.2),
            "top_p": top_p or self.gen_config.get("top_p", 0.95),
            "top_k": top_k or self.gen_config.get("top_k", 30),
            "repetition_penalty": repetition_penalty
            or self.gen_config.get("repetition_penalty", 1.2),
        }

        return run_ehr_inference(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )

