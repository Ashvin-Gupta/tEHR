"""
Baseline 1: Pure LLM Zero-Shot Approach

Uses base Qwen model without any training, just prompting for cancer risk prediction.
"""
import os
import re
import torch
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.baselines.base_baselines import BaseBaseline


class PureLLMBaseline(BaseBaseline):
    """
    Zero-shot LLM baseline for cancer prediction.
    
    Uses base model with simple prompt to extract risk probability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pure LLM baseline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.inference_config = config.get("inference", {})
        
        self.model = None
        self.tokenizer = None

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

        # Counters for probability extraction diagnostics
        self.total_predictions = 0
        self.failed_extractions = 0
    
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
    
    def format_prompt(self, ehr_text: str) -> str:
        """
        Format EHR text with prompt template.
        
        Args:
            ehr_text: Patient EHR text
        
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(ehr_text=ehr_text)
    
    def extract_probability(self, output_text: str, sample_idx: int = None, prompt: str = None) -> float:
        """
        Extract probability from model output.

        Tries multiple strategies to parse probability / binary decision from text.

        Args:
            output_text: Model generated text
            sample_idx: Optional index for debugging failed extractions
            prompt: Optional prompt for debugging failed extractions

        Returns:
            Extracted probability (0.0 to 1.0), or 0.5 if parsing fails
        """
        # Strategy 0: Prefer a leading 0/1 on the first non-empty line
        # Strip whitespace and ALL leading ';' (or similar ...) before interpreting the answer.
        text_stripped = output_text.strip()
        if text_stripped:
            first_line = text_stripped.splitlines()[0].strip()
            # Remove all leading whitespace and any sequence of ';' characters before searching for the answer
            # lstrip with combined " ;" will do whitespace and semicolon
            processed_first_line = first_line.lstrip(" ;")
            # Exact match "0" or "1"
            if processed_first_line in {"0", "1"}:
                return float(processed_first_line)
            # Line starts with 0/1 followed by punctuation / space / text
            if processed_first_line and processed_first_line[0] in {"0", "1"}:
                return float(processed_first_line[0])
            # Patterns like "Answer: 0" or "Answer 1" (let's also do this after lstrip for robustness)
            answer_match = re.search(r'answer[:\s]+([01])\b', processed_first_line.lower())
            if answer_match:
                return float(answer_match.group(1))

        # Strategy 1: Look for explicit probability value
        # Pattern: "0.XX" or "0.XXX" or "0.X"
        prob_patterns = [
            r'\b0\.\d{1,3}\b',  # 0.5, 0.75, 0.123
            r'\b\d+%',  # 50%, 75%
            r'probability[:\s]+([0-9.]+)',
            r'risk[:\s]+([0-9.]+)',
        ]

        for pattern in prob_patterns:
            matches = re.findall(pattern, output_text.lower())
            if matches:
                try:
                    value = float(matches[0].replace('%', ''))
                    if '%' in matches[0]:
                        value = value / 100.0
                    if 0.0 <= value <= 1.0:
                        return value
                except:
                    continue

        # Strategy 2: Look for percentage
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', output_text)
        if percent_match:
            try:
                value = float(percent_match.group(1)) / 100.0
                if 0.0 <= value <= 1.0:
                    return value
            except:
                pass

        # Strategy 3: Look for fraction
        fraction_match = re.search(r'(\d+)/(\d+)', output_text)
        if fraction_match:
            try:
                num = float(fraction_match.group(1))
                den = float(fraction_match.group(2))
                if den > 0:
                    value = num / den
                    if 0.0 <= value <= 1.0:
                        return value
            except:
                pass

        # Default: return 0.5 (neutral)
        self.failed_extractions += 1
        self._log_extraction_failure(output_text, sample_idx=sample_idx, prompt=prompt)
        return 0.5

    def _log_extraction_failure(self, output_text: str, sample_idx: int = None, prompt: str = None):
        """Log failed extraction for debugging. Controlled by config debug flag."""
        debug_cfg = self.config.get('debug', {})
        if not debug_cfg.get('enabled', False):
            print(f"Warning: Could not extract probability from: {output_text[:100]}...")
            return
        # Debug mode: save full details to single file (append each failure)
        log_dir = os.path.join(self.output_dir, 'debug_failed_extractions')
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, "failed_extractions.txt")
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            if sample_idx is not None:
                f.write(f"SAMPLE_INDEX: {sample_idx}\n")
            f.write("RAW_OUTPUT (full):\n")
            f.write(repr(output_text) + "\n\n")
            f.write("RAW_OUTPUT (decoded):\n")
            f.write(output_text + "\n\n")
            if prompt:
                f.write("PROMPT (first 500 chars):\n")
                f.write(prompt[:500] + "...\n\n" if len(prompt) > 500 else prompt + "\n\n")
            f.write("\n")
        if debug_cfg.get('print_first_n', 0) > 0:
            n = debug_cfg['print_first_n']
            if not hasattr(self, '_debug_fail_count'):
                self._debug_fail_count = 0
            if self._debug_fail_count < n:
                self._debug_fail_count += 1
                print(f"\n[DEBUG] Failed extraction #{self._debug_fail_count} (sample_idx={sample_idx})")
                print(f"[DEBUG] Full raw output: {repr(output_text)}")
                if prompt:
                    print(f"[DEBUG] Prompt (first 300 chars): {prompt[:300]}...")
        else:
            print(f"Warning: Could not extract probability from: {output_text[:100]}...")
    
    def predict_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generate predictions for a batch of texts.
        
        Args:
            texts: List of EHR texts
            batch_size: Batch size for inference
        
        Returns:
            Array of predicted probabilities
        """
        self.model.eval()
        probs = []

        # Reset diagnostics for this prediction run
        self.total_predictions = 0
        self.failed_extractions = 0
        if hasattr(self, '_debug_fail_count'):
            delattr(self, '_debug_fail_count')
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"):
                batch_texts = texts[i:i+batch_size]
                batch_prompts = [self.format_prompt(text) for text in batch_texts]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.get('max_length', 8192)
                ).to(self.model.device)
                
                # Generate
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and extract probabilities
                for j, output_ids in enumerate(outputs):
                    # Remove input tokens
                    generated_ids = output_ids[len(inputs['input_ids'][j]):]
                    output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    sample_idx = i + j
                    self.total_predictions += 1
                    prob = self.extract_probability(
                        output_text,
                        sample_idx=sample_idx if self.config.get('debug', {}).get('enabled') else None,
                        prompt=batch_prompts[j] if self.config.get('debug', {}).get('enabled') else None,
                    )
                    probs.append(prob)
        
        # Print simple diagnostics for this call
        if self.total_predictions > 0:
            fail_rate = 100.0 * self.failed_extractions / self.total_predictions
            print(
                f"\nExtraction diagnostics: "
                f"{self.failed_extractions}/{self.total_predictions} "
                f"({fail_rate:.2f}%) outputs fell back to default probability."
            )
        
        return np.array(probs)

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