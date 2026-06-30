"""
Baseline 1: Pure LLM Zero-Shot Approach

Uses base Qwen model without any training, just prompting for cancer risk prediction.

Evaluation uses logit-based probability extraction: the model is run in forward-pass
mode (not generate), and the next-token logits for the tokens "0" and "1" are extracted.
If both logits are below `inference.logit_low_threshold` the prediction defaults to 0.5.
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.baselines.base_baselines import BaseBaseline


class PureLLMBaseline(BaseBaseline):
    """
    Zero-shot LLM baseline using logit-based probability extraction.
    
    For each patient prompt, the model performs a single forward pass and the
    next-token logits for "0" and "1" are extracted.  The probability of the
    positive class is softmax({logit_0, logit_1})[1].  If both logits are below
    `inference.logit_low_threshold` (default -10.0) the prediction is 0.5.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pure LLM baseline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.inference_config = config.get("inference", {})
        self.logit_low_threshold = float(
            self.inference_config.get("logit_low_threshold", -10.0)
        )
        
        self.model = None
        self.tokenizer = None
        self._token_id_0: int | None = None
        self._token_id_1: int | None = None

        # Load prompt template from external file so we can easily experiment
        # with different prompts without changing code.
        prompt_cfg = config.get("prompt", {})
        prompt_path = prompt_cfg.get(
            "filepath",
            "src/resources/pure_llm_prompt.txt",
        )
        if not os.path.isabs(prompt_path):
            # Resolve relative to project root (assume CWD is repo root)
            prompt_path = os.path.join(os.getcwd(), prompt_path)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Counters for low-logit fallback diagnostics
        self.total_predictions = 0
        self.low_logit_fallbacks = 0
    
    def load_model(self):
        """Load base Qwen model (no pretraining, no LoRA)."""
        print("\n" + "=" * 80)
        print("Loading base Qwen model...")
        print("=" * 80)
        
        model_name = self.model_config.get('model_name', 'unsloth/Qwen2.5-7B-Instruct')
        max_length = self.model_config.get('max_length', 8192)
        load_in_4bit = self.model_config.get('load_in_4bit', True)
        
        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - 4-bit quantization: {load_in_4bit}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto"
        )
        
        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")

        # Cache the token IDs for the literal strings "0" and "1".
        # Most tokenizers map these to a single token; if multiple tokens are
        # returned we take the first one (covers leading-space variants).
        self._token_id_0 = self.tokenizer.encode("0", add_special_tokens=False)[0]
        self._token_id_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]
        print(
            f"  - Token IDs: '0' → {self._token_id_0}, '1' → {self._token_id_1}"
        )
    
    def format_prompt(self, ehr_text: str) -> str:
        """
        Format EHR text with prompt template.
        
        Args:
            ehr_text: Patient EHR text
        
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(ehr_text=ehr_text)

    def _logits_to_prob(self, logit_0: float, logit_1: float) -> float:
        """
        Convert next-token logits for "0" and "1" to a positive-class probability.

        If both logits are below `logit_low_threshold` the model is not expressing
        a clear binary preference, so we return 0.5.

        Args:
            logit_0: Raw next-token logit for the "0" token.
            logit_1: Raw next-token logit for the "1" token.

        Returns:
            Probability in [0, 1] for the positive class ("1").
        """
        if max(logit_0, logit_1) < self.logit_low_threshold:
            self.low_logit_fallbacks += 1
            return 0.5
        pair = torch.tensor([logit_0, logit_1], dtype=torch.float32)
        return float(torch.softmax(pair, dim=0)[1].item())

    def predict_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Extract positive-class probabilities via logit-based inference.

        For each prompt we do a single forward pass (no generation) and read the
        next-token logits for the tokens "0" and "1".

        Args:
            texts: List of EHR texts.
            batch_size: Batch size for inference.
        
        Returns:
            Float32 array of positive-class probabilities, one per input text.
        """
        self.model.eval()
        probs = []

        # Reset diagnostics
        self.total_predictions = 0
        self.low_logit_fallbacks = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Logit inference"):
                batch_texts = texts[i:i + batch_size]
                batch_prompts = [self.format_prompt(t) for t in batch_texts]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.get("max_length", 8192),
                ).to(self.model.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)

                # Next-token logits at the last real (non-padding) position per sample.
                # outputs.logits shape: (batch, seq_len, vocab_size)
                next_token_logits = outputs.logits  # (B, T, V)
                seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # (B,)
                batch_sz = next_token_logits.size(0)
                last_logits = next_token_logits[
                    range(batch_sz), seq_lengths, :
                ]  # (B, V)

                logit_0 = last_logits[:, self._token_id_0].cpu().float()
                logit_1 = last_logits[:, self._token_id_1].cpu().float()

                for j in range(batch_sz):
                    self.total_predictions += 1
                    prob = self._logits_to_prob(
                        logit_0[j].item(), logit_1[j].item()
                    )
                    probs.append(prob)

        if self.total_predictions > 0:
            fallback_rate = 100.0 * self.low_logit_fallbacks / self.total_predictions
            print(
                f"\nLogit diagnostics: "
                f"{self.low_logit_fallbacks}/{self.total_predictions} "
                f"({fallback_rate:.2f}%) samples fell back to 0.5 "
                f"(both logits < {self.logit_low_threshold})."
            )

        return np.array(probs, dtype=np.float32)

    def predict(self, dataset) -> np.ndarray:
        """
        Run zero-shot prompting over the dataset and return probabilities
        for the positive class.

        Note: `BaseBaseline.evaluate_split()` builds labels by iterating the
        dataset and skipping `None` items. We mirror that ordering here so
        labels and probabilities align.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Did you call load_model()?")

        texts: List[str] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item is None:
                continue
            text = item.get("text")
            if text is None:
                continue
            texts.append(text)

        if not texts:
            return np.array([], dtype=np.float32)

        batch_size = int(self.inference_config.get("batch_size", 4))
        probs = self.predict_batch(texts, batch_size=batch_size)
        return probs.astype(np.float32)