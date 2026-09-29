"""
Experiment: zero_shot — prompt the LLM with no task fine-tuning.

Two inference modes (``inference.mode``):

- ``logit_01``: one forward pass; P(1) = softmax of next-token logits for ``0``/``1``.
- ``generate_probability``: generate a short completion and parse a probability in ``[0, 1]``.
"""
import os
import re
import math
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.baselines.base import BaseBaseline
from src.evaluations.evaluations import compute_binary_metrics, write_split_artifacts
from src.evaluations.eval_resume import (
    load_eval_partial,
    maybe_flush_partial,
    parse_eval_shard,
    partial_pred_path,
    shard_bounds,
    shard_result_path,
    try_merge_shards,
    write_shard_file,
)
from src.training.training_utils import resolve_binary_label_token_ids


_PROB_RE = re.compile(
    r"(?<!\d)((?:0?\.\d{1,4})|1\.0{1,4}|0|1)(?!\d)"
)


class ZeroShotBaseline(BaseBaseline):
    """
    Zero-shot LLM prompting.

    ``inference.mode`` (see ``configs/experiments/zero_shot.yaml``):

    - ``generate_probability`` (default in YAML): generate a short completion
      and parse a probability in ``[0, 1]``.
    - ``logit_01``: one forward pass; P(1) = softmax of next-token logits for
      ``0`` / ``1``. If both logits are below ``logit_low_threshold``, P=0.5.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pure LLM baseline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.inference_config = config.get("inference", {})
        self.debug_config = config.get("debug", {})
        self.logit_low_threshold = float(
            self.inference_config.get("logit_low_threshold", -10.0)
        )
        self.inference_mode = str(
            self.inference_config.get("mode", "logit_01")
        ).lower()
        self.max_new_tokens = int(self.inference_config.get("max_new_tokens", 8))
        
        self.model = None
        self.tokenizer = None
        self._token_id_0: int | None = None
        self._token_id_1: int | None = None

        # Load prompt template from external file so we can easily experiment
        # with different prompts without changing code.
        prompt_cfg = config.get("prompt", {})
        default_prompt = (
            "src/resources/probability_task_prompt.txt"
            if self.inference_mode == "generate_probability"
            else "src/resources/binary_task_prompt.txt"
        )
        prompt_path = prompt_cfg.get("filepath", default_prompt)
        if not os.path.isabs(prompt_path):
            # Resolve relative to project root (assume CWD is repo root)
            prompt_path = os.path.join(os.getcwd(), prompt_path)
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print(f"  - Prompt template: {prompt_path}")
        print(f"  - Inference mode: {self.inference_mode}")
        print(f"  - Debug prompts: {bool(self.debug_config.get('enabled', False))}")
        # Instruct models need the chat template so next-token mass is not
        # sitting on special tokens.
        self.use_chat_template = bool(
            prompt_cfg.get(
                "use_chat_template",
                self.model_config.get("use_chat_template", True),
            )
        )
        default_prefill = "Answer:"
        self.assistant_prefill = str(
            prompt_cfg.get(
                "assistant_prefill",
                self.model_config.get("assistant_prefill", default_prefill),
            )
        )

        # Counters for low-logit / parse fallback diagnostics
        self.total_predictions = 0
        self.low_logit_fallbacks = 0
        self.parse_fallbacks = 0
        self._debug_printed = 0
        self._prompt_overhead_tokens: int | None = None
        self._eos_token_ids: List[int] | None = None
        self._suppress_tokens: List[int] | None = None
    
    def load_model(self):
        """Load the base causal LM (no pretraining, no LoRA)."""
        print("\n" + "=" * 80)
        print("Loading base model...")
        print("=" * 80)
        
        model_name = self.model_config.get('model_name', 'unsloth/Qwen2.5-7B-Instruct')
        max_length = self.model_config.get('max_length', 8192)
        load_in_4bit = self.model_config.get('load_in_4bit', True)
        
        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - 4-bit quantization: {load_in_4bit}")
        
        use_unsloth = self.model_config.get("use_unsloth", True)

        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
                device_map="auto",
            )
            if self.inference_mode == "generate_probability":
                FastLanguageModel.for_inference(self.model)
        else:
            from src.training.model_loader import load_hf_model
            self.model, self.tokenizer = load_hf_model(
                model_name=model_name,
                max_seq_length=max_length,
                load_in_4bit=load_in_4bit,
                local_rank=0,
            )

        # Decoder-only generation must left-pad.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._eos_token_ids = self._resolve_eos_token_ids()
        self._suppress_tokens = self._resolve_suppress_tokens()

        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")

        # Cache vocab IDs for answer tokens "0" / "1" (not encode(...)[0],
        # which on Mistral/Llama is the shared ▁ spacepiece for both digits).
        self._token_id_0, self._token_id_1 = resolve_binary_label_token_ids(
            self.tokenizer
        )
        print(
            f"  - Token IDs: '0' → {self._token_id_0}, '1' → {self._token_id_1}"
        )
        print(
            "  - Token pieces: "
            f"'0' → {self.tokenizer.convert_ids_to_tokens(self._token_id_0)!r}, "
            f"'1' → {self.tokenizer.convert_ids_to_tokens(self._token_id_1)!r}"
        )
        # How "0.25" tokenises — relevant for generate_probability parsing.
        sample_ids = self.tokenizer.encode("0.25", add_special_tokens=False)
        sample_pieces = [self.tokenizer.convert_ids_to_tokens(i) for i in sample_ids]
        print(f"  - encode('0.25') → ids={sample_ids} pieces={sample_pieces}")
        self._prompt_overhead_tokens = self._measure_prompt_overhead()
        print(f"  - Prompt overhead ≈ {self._prompt_overhead_tokens} tokens")
        print(f"  - use_chat_template = {self.use_chat_template}")
        print(f"  - assistant_prefill = {self.assistant_prefill!r}")
        print(f"  - padding_side = {self.tokenizer.padding_side}")
        print(f"  - eos_token_ids = {self._eos_token_ids}")
        print(f"  - suppress_tokens = {self._suppress_tokens}")

    def _resolve_eos_token_ids(self) -> List[int]:
        """Include eos_token_id plus any tokenizer-specific end-of-turn id."""
        ids: List[int] = []
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.append(int(eos))
        eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if (
            isinstance(eot, int)
            and eot >= 0
            and self.tokenizer.convert_ids_to_tokens(eot) == "<end_of_turn>"
            and eot not in ids
        ):
            ids.append(int(eot))
        return ids

    def _resolve_suppress_tokens(self) -> List[int]:
        """
        Suppress pad during greedy decode so generation cannot collapse to pads.
        Never suppress eos / end-of-turn.
        """
        pad = self.tokenizer.pad_token_id
        if pad is None:
            return []
        eos_set = set(self._eos_token_ids or [])
        if int(pad) in eos_set:
            return []
        return [int(pad)]

    def _measure_prompt_overhead(self) -> int:
        """Tokens used by the prompt wrapper with an empty EHR (+ chat template)."""
        dummy = self.format_prompt("", truncate_ehr=False)
        ids = self.tokenizer.encode(dummy, add_special_tokens=True)
        return len(ids) + 8  # small safety margin

    def _truncate_ehr(self, ehr_text: str) -> str:
        """
        Drop oldest clinical events so the full prompt (incl. ``Answer:``) fits
        in ``max_length``.  HF's default right-truncation would otherwise chop
        off the answer suffix and leave the model predicting mid-EHR.
        """
        max_length = int(self.model_config.get("max_length", 8192))
        overhead = self._prompt_overhead_tokens or 64
        budget = max_length - overhead
        if budget <= 64:
            return ehr_text

        marker = " <HEADER_SPLIT> "
        if marker in ehr_text:
            header, body = ehr_text.split(marker, 1)
        else:
            header, body = "", ehr_text

        header_ids = (
            self.tokenizer.encode(header, add_special_tokens=False) if header else []
        )
        body_budget = budget - len(header_ids)
        if body_budget <= 0:
            return header

        body_ids = self.tokenizer.encode(body, add_special_tokens=False)
        if len(body_ids) <= body_budget:
            return ehr_text

        truncated_body = self.tokenizer.decode(
            body_ids[-body_budget:], skip_special_tokens=True
        )
        if header:
            return header + marker + truncated_body
        return truncated_body

    def format_prompt(self, ehr_text: str, truncate_ehr: bool = True) -> str:
        """
        Format EHR text with prompt template, optionally wrap in chat template.

        Cue ``Answer:`` after the chat turn so the next token is the score
        rather than prose.
        """
        if truncate_ehr and self.tokenizer is not None:
            ehr_text = self._truncate_ehr(ehr_text)

        task_desc = self.config.get("task", {}).get("description", "")
        try:
            user_text = self.prompt_template.format(ehr_text=ehr_text, task=task_desc)
        except KeyError:
            user_text = self.prompt_template.format(ehr_text=ehr_text)

        answer_suffix = "Answer:"
        stripped = user_text.rstrip()
        has_answer_cue = stripped.endswith(answer_suffix)
        if has_answer_cue and self.assistant_prefill:
            # Move cue out of the user turn; re-attach after chat wrap.
            user_text = stripped[: -len(answer_suffix)].rstrip() + "\n"
        elif not has_answer_cue and not self.assistant_prefill:
            # Ensure the user turn still asks for an Answer: line.
            user_text = stripped + "\n\n" + answer_suffix + "\n"

        if (
            self.use_chat_template
            and self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None)
        ):
            try:
                wrapped = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return wrapped + self.assistant_prefill
            except Exception as exc:
                print(f"  - WARNING: apply_chat_template failed ({exc}); using raw prompt")
                return user_text + self.assistant_prefill
        if self.assistant_prefill and not user_text.rstrip().endswith(self.assistant_prefill):
            return user_text + self.assistant_prefill
        return user_text

    def _preview_prompt(self, prompt: str, max_chars: int = 1200) -> str:
        """Keep head + tail of a long prompt so the Answer: suffix stays visible."""
        if len(prompt) <= max_chars:
            return prompt
        head = max_chars // 2
        tail = max_chars - head
        return (
            prompt[:head]
            + f"\n... [{len(prompt) - max_chars} chars omitted] ...\n"
            + prompt[-tail:]
        )

    def _maybe_debug_example(
        self,
        *,
        index: int,
        prompt: str,
        n_tokens: int,
        truncated: bool,
        logit_0: float,
        logit_1: float,
        prob: float,
        top_tokens: List[str],
        generated: Optional[str] = None,
    ) -> None:
        if not self.debug_config.get("enabled", False):
            return
        limit = int(self.debug_config.get("print_first_n", 5))
        if self._debug_printed >= limit:
            return
        self._debug_printed += 1
        task_desc = self.config.get("task", {}).get("description", "")
        print("\n" + "=" * 80, flush=True)
        print(f"DEBUG prompt example #{self._debug_printed}/{limit} (dataset idx={index})", flush=True)
        print(f"  task.description = {task_desc!r}", flush=True)
        print(f"  tokens = {n_tokens}  truncated = {truncated}", flush=True)
        if generated is not None:
            print(f"  generated = {generated!r}  parsed P = {prob:.4f}", flush=True)
            if top_tokens:
                print(f"  gen detail: {top_tokens}", flush=True)
        else:
            print(f"  logit_0 = {logit_0:.4f}  logit_1 = {logit_1:.4f}  P(1) = {prob:.4f}", flush=True)
            print(f"  top next-tokens: {top_tokens}", flush=True)
        print("-" * 80, flush=True)
        print(self._preview_prompt(prompt), flush=True)
        print("=" * 80 + "\n", flush=True)

    @staticmethod
    def _parse_probability(text: str) -> Optional[float]:
        """Extract the first plausible [0, 1] probability from model output."""
        if not text:
            return None
        for match in _PROB_RE.finditer(text.strip()):
            try:
                val = float(match.group(1))
            except ValueError:
                continue
            if 0.0 <= val <= 1.0:
                return float(val)        # Percent form, e.g. 23%
        pct = re.search(r"(\d{1,3})\s*%", text)
        if pct:
            val = float(pct.group(1)) / 100.0
            if 0.0 <= val <= 1.0:
                return val
        return None

    def _logits_to_prob(self, logit_0: float, logit_1: float) -> float:
        """
        Convert next-token logits for "0" and "1" to a positive-class probability.

        If both logits are below `logit_low_threshold` the model is not expressing
        a clear binary preference, so we return 0.5.
        """
        if not (math.isfinite(logit_0) and math.isfinite(logit_1)):
            self.low_logit_fallbacks += 1
            return 0.5
        if max(logit_0, logit_1) < self.logit_low_threshold:
            self.low_logit_fallbacks += 1
            return 0.5
        pair = torch.tensor([logit_0, logit_1], dtype=torch.float32)
        return float(torch.softmax(pair, dim=0)[1].item())

    def _tokenize_prompts(self, batch_prompts: List[str], max_length: int):
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)
        if inputs["input_ids"].size(1) > max_length:
            inputs["input_ids"] = inputs["input_ids"][:, -max_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_length:]
        return inputs

    def predict_batch_logit_01(
        self,
        texts: List[str],
        batch_size: int = 4,
        *,
        resume_probs: Optional[List[float]] = None,
        partial_path: Optional[str] = None,
    ) -> np.ndarray:
        """Forward-pass scoring via softmax over next-token logits for 0/1."""
        self.model.eval()
        probs: List[float] = list(resume_probs or [])
        n_expected = len(probs) + len(texts)
        self.total_predictions = 0
        self.low_logit_fallbacks = 0
        self._debug_printed = 0
        max_length = int(self.model_config.get("max_length", 8192))

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Logit inference"):
                batch_texts = texts[i:i + batch_size]
                batch_prompts = [self.format_prompt(t) for t in batch_texts]
                inputs = self._tokenize_prompts(batch_prompts, max_length)

                outputs = self.model(**inputs)

                next_token_logits = outputs.logits
                attention_mask = inputs["attention_mask"]
                last_indices = attention_mask.long().cumsum(dim=1).argmax(dim=1)
                batch_sz = next_token_logits.size(0)
                # float32 for stable softmax over the two digit logits
                last_logits = next_token_logits[
                    range(batch_sz), last_indices, :
                ].float()

                logit_0 = last_logits[:, self._token_id_0].cpu()
                logit_1 = last_logits[:, self._token_id_1].cpu()

                for j in range(batch_sz):
                    self.total_predictions += 1
                    l0 = float(logit_0[j].item())
                    l1 = float(logit_1[j].item())
                    prob = self._logits_to_prob(l0, l1)
                    probs.append(prob)

                    n_tokens = int(attention_mask[j].sum().item())
                    top_ids = torch.topk(last_logits[j], k=5).indices.tolist()
                    last_tok = int(inputs["input_ids"][j, last_indices[j]].item())
                    top_tokens = [
                        f"last_tok={self.tokenizer.decode([last_tok], skip_special_tokens=False)!r}"
                    ] + [
                        repr(self.tokenizer.decode([tid], skip_special_tokens=False))
                        for tid in top_ids
                    ]
                    self._maybe_debug_example(
                        index=len(probs) - 1,
                        prompt=batch_prompts[j],
                        n_tokens=n_tokens,
                        truncated=n_tokens >= max_length,
                        logit_0=l0,
                        logit_1=l1,
                        prob=prob,
                        top_tokens=top_tokens,
                    )
                maybe_flush_partial(partial_path, probs, n_expected)

        maybe_flush_partial(partial_path, probs, n_expected, force=True)

        if self.total_predictions > 0:
            fallback_rate = 100.0 * self.low_logit_fallbacks / self.total_predictions
            print(
                f"\nLogit diagnostics: "
                f"{self.low_logit_fallbacks}/{self.total_predictions} "
                f"({fallback_rate:.2f}%) samples fell back to 0.5 "
                f"(both logits < {self.logit_low_threshold} or non-finite)."
            )
        return np.array(probs, dtype=np.float32)

    def predict_batch_generate_probability(
        self,
        texts: List[str],
        batch_size: int = 1,
        *,
        resume_probs: Optional[List[float]] = None,
        partial_path: Optional[str] = None,
    ) -> np.ndarray:
        """Generate a short answer and parse a 2-decimal probability in [0, 1]."""
        self.model.eval()
        probs: List[float] = list(resume_probs or [])
        n_expected = len(probs) + len(texts)
        self.total_predictions = 0
        self.parse_fallbacks = 0
        self._debug_printed = 0
        max_length = int(self.model_config.get("max_length", 8192))
        # Generation with variable-length prompts is simplest at batch_size=1
        # when left-padded; keep configurable but default small.
        batch_size = max(1, int(batch_size))
        eos_token_id = self._eos_token_ids or self.tokenizer.eos_token_id
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
        if self._suppress_tokens:
            gen_kwargs["suppress_tokens"] = self._suppress_tokens

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generate P"):
                batch_texts = texts[i:i + batch_size]
                batch_prompts = [self.format_prompt(t) for t in batch_texts]
                inputs = self._tokenize_prompts(batch_prompts, max_length)
                prompt_width = inputs["input_ids"].shape[1]
                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

                generated = self.model.generate(**inputs, **gen_kwargs)

                for j in range(len(batch_prompts)):
                    self.total_predictions += 1
                    # Always slice from padded width (correct for left- and right-pad).
                    new_ids = generated[j][prompt_width:]
                    # Drop leading pads if any slipped through.
                    pad_id = self.tokenizer.pad_token_id
                    ids_list = new_ids.tolist()
                    if pad_id is not None:
                        ids_list = [t for t in ids_list if t != pad_id]
                    gen_text = self.tokenizer.decode(
                        ids_list, skip_special_tokens=True
                    ).strip()
                    gen_raw = self.tokenizer.decode(
                        new_ids, skip_special_tokens=False
                    )

                    parsed = self._parse_probability(gen_text)
                    if parsed is None:
                        self.parse_fallbacks += 1
                        prob = 0.5
                    else:
                        prob = parsed
                    probs.append(prob)

                    n_tokens = int(prompt_lens[j])
                    self._maybe_debug_example(
                        index=len(probs) - 1,
                        prompt=batch_prompts[j],
                        n_tokens=n_tokens,
                        truncated=n_tokens >= max_length,
                        logit_0=0.0,
                        logit_1=0.0,
                        prob=prob,
                        top_tokens=[
                            f"new_ids={new_ids.tolist()}",
                            f"raw={gen_raw!r}",
                        ],
                        generated=gen_text,
                    )
                maybe_flush_partial(partial_path, probs, n_expected)

        maybe_flush_partial(partial_path, probs, n_expected, force=True)

        if self.total_predictions > 0:
            rate = 100.0 * self.parse_fallbacks / self.total_predictions
            print(
                f"\nGenerate-P diagnostics: "
                f"{self.parse_fallbacks}/{self.total_predictions} "
                f"({rate:.2f}%) failed to parse and fell back to 0.5."
            )
        return np.array(probs, dtype=np.float32)

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        **resume_kw,
    ) -> np.ndarray:
        if self.inference_mode == "generate_probability":
            return self.predict_batch_generate_probability(
                texts, batch_size=batch_size, **resume_kw
            )
        if self.inference_mode in {"logit_01", "logit", "binary_logit"}:
            return self.predict_batch_logit_01(
                texts, batch_size=batch_size, **resume_kw
            )
        raise ValueError(
            f"Unknown inference.mode={self.inference_mode!r}. "
            "Use 'logit_01' or 'generate_probability'."
        )

    def _collect_texts_labels(self, dataset) -> Tuple[List[str], List[int]]:
        texts: List[str] = []
        labels: List[int] = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item is None:
                continue
            text = item.get("text")
            if text is None:
                continue
            texts.append(text)
            label_value = item["label"].item()
            labels.append(0 if label_value == 0 else 1)
        return texts, labels

    def _predict_with_resume(self, texts: List[str], tag: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Did you call load_model()?")
        if not texts:
            return np.array([], dtype=np.float32)
        path = partial_pred_path(self.output_dir, tag)
        done = load_eval_partial(path, len(texts))
        remaining = texts[len(done):]
        batch_size = int(self.inference_config.get("batch_size", 4))
        if not remaining:
            print(f"  All {len(texts)} {tag} predictions already flushed.", flush=True)
            return np.array(done, dtype=np.float32)
        return self.predict_batch(
            remaining,
            batch_size=batch_size,
            resume_probs=done,
            partial_path=path,
        ).astype(np.float32)

    def predict(self, dataset) -> np.ndarray:
        """
        Run zero-shot prompting over the dataset and return probabilities
        for the positive class.

        Note: `BaseBaseline.evaluate_split()` builds labels by iterating the
        dataset and skipping `None` items. We mirror that ordering here so
        labels and probabilities align.
        """
        texts, _labels = self._collect_texts_labels(dataset)
        return self._predict_with_resume(texts, "predict")

    def evaluate_split(self, dataset, split_name: str) -> Dict[str, float]:
        """Score one split, flushing partials and optionally a ``EVAL_SHARD`` slice."""
        texts, labels = self._collect_texts_labels(dataset)
        n_total = len(texts)
        max_n = int((self.debug_config or {}).get("max_samples") or 0)
        if max_n > 0 and max_n < n_total:
            texts, labels = texts[:max_n], labels[:max_n]
            n_total = len(texts)
            print(f"  debug.max_samples={max_n}", flush=True)
        shard = parse_eval_shard()
        if shard:
            index, n_shards = shard
            start, end = shard_bounds(n_total, index, n_shards)
            texts = texts[start:end]
            labels = labels[start:end]
            tag = f"{split_name}_shard_{index}of{n_shards}"
            print(
                f"Zero-shot shard {index}/{n_shards}: "
                f"indices [{start}, {end}) of {n_total}",
                flush=True,
            )
        else:
            index, n_shards, start, end = 0, 1, 0, n_total
            tag = split_name

        probs = self._predict_with_resume(texts, tag)
        if len(probs) != len(labels):
            raise RuntimeError(
                f"{tag}: got {len(probs)} probs for {len(labels)} labels"
            )

        if shard:
            shard_path = shard_result_path(
                self.output_dir, split_name, index, n_shards
            )
            write_shard_file(
                shard_path,
                split_name=split_name,
                index=index,
                n_shards=n_shards,
                start=start,
                end=end,
                n_total=n_total,
                labels=labels,
                probs=[float(p) for p in probs.tolist()],
            )
            partial = partial_pred_path(self.output_dir, tag)
            if os.path.isfile(partial):
                os.remove(partial)
            merged = try_merge_shards(
                self.output_dir, split_name, n_shards, n_total
            )
            if merged is None:
                print(
                    "  This shard is done. Submit the other EVAL_SHARD jobs; "
                    "the last one to finish writes held_out_results.json.",
                    flush=True,
                )
                return compute_binary_metrics(np.array(labels), probs)
            labels, probs_list = merged
            metrics = write_split_artifacts(
                self.output_dir, split_name, np.array(labels), np.array(probs_list)
            )
            return metrics

        metrics = write_split_artifacts(
            self.output_dir, split_name, np.array(labels), probs
        )
        partial = partial_pred_path(self.output_dir, tag)
        if os.path.isfile(partial):
            os.remove(partial)
        return metrics

    def run(self, resume_from_checkpoint: str | None = None):
        """Score held_out only. Zero-shot has no training and does not need val."""
        self.setup_wandb()
        self.load_model()
        datasets = self.load_datasets(splits=["held_out"])
        shard = parse_eval_shard()
        extra = f" EVAL_SHARD={shard[0]}/{shard[1]}" if shard else ""
        print(
            "Zero-shot: scoring test split (held_out) only; "
            f"skipping tuning/val.{extra}",
            flush=True,
        )
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")
        return None, test_metrics