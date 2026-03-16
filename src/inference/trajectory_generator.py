"""
High-level interface for generating future trajectories from EHR records
using a pretrained LLM checkpoint, closely mirroring the Pure LLM baseline
but adapted to the MIMIC dataset.
"""
from typing import Dict, Any, Optional

import numpy as np
from unsloth import FastLanguageModel

from src.training.model_loader import load_pretrained_lora_model
from src.inference.generator import run_ehr_inference
from src.baselines.utils import load_baseline_config, load_datasets
from src.evaluations.baseline_metrics import (
    compute_baseline_metrics,
    plot_all_curves,
    save_results,
    print_results,
)


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
    Wrapper around a pretrained LLM (e.g., LoRA‑fine‑tuned) for trajectory generation.
    Uses a simple prompt template that inserts the full patient EHR text.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary with keys:
                - model:
                    - checkpoint_dir: path to pretrained checkpoint (LoRA or full model)
                    - max_length: max context length
                    - load_in_4bit: bool
                    - local_rank: optional device index
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

        # Load the pretrained checkpoint instead of a fresh base model.
        self.model, self.tokenizer = load_pretrained_lora_model(
            pretrained_checkpoint=checkpoint_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            local_rank=local_rank,
        )

    def _truncate_ehr_text_for_generation(self, ehr_text: str, max_new_tokens: int) -> str:
        """
        If the full prompt (instruction + EHR text) would exceed the model's
        maximum sequence length once we add `max_new_tokens`, truncate the
        EHR part while preserving the header and the most recent events.
        """
        max_seq_len = int(self.model_config.get("max_length", 8192))
        if max_seq_len <= max_new_tokens:
            return ehr_text

        template = self.prompt_config.get("template", DEFAULT_PROMPT_TEMPLATE)
        split_token = "<HEADER_SPLIT>"

        # Lightweight iterative truncation; number of examples is small so this
        # is acceptable.
        truncated = ehr_text
        for _ in range(10):
            prompt = template.format(ehr_text=truncated)
            input_ids = self.tokenizer(
                [prompt],
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"][0]

            if input_ids.shape[0] + max_new_tokens <= max_seq_len:
                return truncated

            # Need to shrink: keep header (before split token) and tail of body.
            if split_token in truncated:
                header_part, body_part = truncated.split(split_token, 1)
                # If body is already very short, fall back to basic tail truncation.
                if len(body_part) < 200:
                    truncated = truncated[-len(truncated) // 2 :]
                    continue
                # Keep the last half of the body (most recent events).
                keep_len = len(body_part) // 2
                body_part = body_part[-keep_len:]
                truncated = header_part + split_token + body_part
            else:
                # No explicit split marker; keep the tail of the whole sequence.
                if len(truncated) < 200:
                    # Already short enough; stop iterating.
                    return truncated
                truncated = truncated[-len(truncated) // 2 :]

        return truncated

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
        max_new_tokens = max_new_tokens or self.gen_config.get("max_new_tokens", 256)

        # Truncate the EHR text if prompt + generation would exceed the model limit,
        # preserving header and most recent events where possible.
        truncated_ehr_text = self._truncate_ehr_text_for_generation(
            ehr_text=ehr_text,
            max_new_tokens=max_new_tokens,
        )

        prompt = self._build_prompt(truncated_ehr_text, extra_instruction=extra_instruction)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
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


class MIMICTrajectoryGenerationRunner:
    """
    End‑to‑end runner that:
      1) Loads the MIMIC dataset using the same config style as the Pure LLM baseline.
      2) For a small number of patients, takes the translated trajectory text.
      3) Feeds it to a pretrained checkpoint to *continue* the trajectory instead of
         predicting a 0/1 label.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config["data"]
        self.generator = EHRTrajectoryGenerator(config)

        # Optional generation‑control & benchmarking settings in config:
        gen_cfg = config.get("trajectory_generation", {})
        self.num_examples_cfg = gen_cfg.get("num_examples", None)
        self.repetitions_cfg = int(gen_cfg.get("repetitions", 1))
        # Keywords in generated text that we interpret as predicting death.
        self.positive_keywords = [
            kw.lower() for kw in gen_cfg.get("positive_keywords", ["death", "meds_death"])
        ]

    def _sequence_contains_positive_keyword(self, text: str) -> bool:
        """
        Check whether a generated continuation contains any of the configured
        'positive' keywords signalling a death event.
        """
        lower_text = text.lower()
        return any(kw in lower_text for kw in self.positive_keywords)

    def run(
        self,
        split: str = "tuning",
        num_examples: Optional[int] = None,
        repetitions: Optional[int] = None,
        benchmark: bool = False,
    ):
        """
        Generate and print continuations for a few patient trajectories.

        Args:
            split: Dataset split to sample from (e.g. tuning, held_out).
            num_examples: Overrides the config value if provided.
            repetitions: Number of independent generations per datapoint
                (e.g. 5 -> 5 samples per patient). Overrides config if provided.
            benchmark: If True, do not print every continuation; instead compute
                per‑patient scores based on how often a 'death' token appears,
                then compute ROC/AUPR against true labels.
        """
        # Resolve effective settings from CLI arguments or config.
        if num_examples is None:
            num_examples = self.num_examples_cfg
        if repetitions is None:
            repetitions = self.repetitions_cfg

        # Re‑use the same dataset loading logic as the Pure LLM baseline.
        datasets = load_datasets(
            self.data_config,
            splits=[split],
            format="text",
        )
        dataset = datasets[split]

        num_examples = min(num_examples, len(dataset))
        print(
            f"\nGenerating trajectory continuations for {num_examples} examples "
            f"from split '{split}' with {repetitions} repetitions per example"
        )

        labels_all = []
        scores_all = []  # fraction of repetitions where a positive keyword appears

        for i in range(num_examples):
            sample = dataset[i]
            if sample is None:
                continue

            ehr_text = sample["text"]
            label = sample["label"].item() if hasattr(sample["label"], "item") else sample["label"]

            if not benchmark:
                print("\n" + "=" * 80)
                print(f"EXAMPLE {i}  (label={label})")
                print("=" * 80)

                # Show a truncated view of the input trajectory.
                print("\n[INPUT TRAJECTORY - first 1000 chars]\n")
                print(ehr_text[:1000])
                if len(ehr_text) > 1000:
                    print("...\n")

            # Generate one or more independent continuations from the pretrained checkpoint.
            positive_count = 0
            for r in range(repetitions):
                generated = self.generator.generate(ehr_text=ehr_text)

                if self._sequence_contains_positive_keyword(generated):
                    positive_count += 1

                if not benchmark:
                    print(f"\n[GENERATED CONTINUATION #{r + 1}]\n")
                    print(generated)

            # For benchmarking: score is fraction of repetitions with death keyword(s).
            if benchmark:
                labels_all.append(int(label))
                score = positive_count / float(repetitions) if repetitions > 0 else 0.0
                scores_all.append(score)

            # ------------------------------------------------------------------
            # Ground truth segment: first N events *after* the prediction time.
            #
            # We reconstruct this from the full MIMIC record + ACES row rather
            # than from `ehr_text`, because `ehr_text` has already been
            # truncated to events up to the prediction time.
            # ------------------------------------------------------------------
            max_events = self.generator.gen_config.get("max_new_tokens", 256)

            # Access the underlying MIMICUnifiedDataset internals to get:
            #   - the full token/timestamp trajectory
            #   - the prediction_time for this ACES example
            try:
                example_row = dataset.examples[i]
                subject_id = example_row.get("subject_id")
                records_for_subject = dataset.subject_to_records.get(subject_id)
            except Exception:
                records_for_subject = None

            ground_truth_segment = ""
            if records_for_subject:
                record = records_for_subject[0]
                tokens_full = record.get("tokens", [])
                timestamps_full = record.get("timestamps", [])
                pred_time = example_row.get("prediction_time")

                if pred_time is not None and len(timestamps_full) > 0:
                    first_ts = timestamps_full[0]

                    # Case 1: numeric timestamps
                    if isinstance(first_ts, (int, float)):
                        import pandas as pd  # local import to avoid top‑level dependency

                        numeric_pred_time = None
                        if isinstance(pred_time, (int, float)):
                            numeric_pred_time = float(pred_time)
                        elif isinstance(pred_time, pd.Timestamp):
                            try:
                                numeric_pred_time = float(pred_time.timestamp())
                            except Exception:
                                numeric_pred_time = None

                        if numeric_pred_time is not None:
                            future_tokens = [
                                t
                                for t_idx, t in enumerate(tokens_full)
                                if timestamps_full[t_idx] > numeric_pred_time
                            ]
                        else:
                            future_tokens = []

                    # Case 2: pandas.Timestamp timestamps
                    else:
                        import pandas as pd  # local import to avoid top‑level dependency

                        ts_pred_time = None
                        if isinstance(pred_time, pd.Timestamp):
                            ts_pred_time = pred_time
                        elif isinstance(pred_time, (int, float)):
                            try:
                                ts_pred_time = pd.to_datetime(pred_time, unit="s")
                            except Exception:
                                ts_pred_time = None

                        if ts_pred_time is not None:
                            future_tokens = [
                                t
                                for t_idx, t in enumerate(tokens_full)
                                if timestamps_full[t_idx] > ts_pred_time
                            ]
                        else:
                            future_tokens = []

                    # Take the first `max_events` tokens after prediction time.
                    future_tokens = future_tokens[:max_events]

                    if future_tokens:
                        # Translate using the same token‑to‑text mapping as the dataset.
                        string_codes = [
                            dataset.id_to_token_map.get(tid, "") for tid in future_tokens
                        ]
                        translated_phrases = []
                        for code in string_codes:
                            phrase = dataset._translate_token(str(code))
                            if phrase:
                                translated_phrases.append(phrase)
                        ground_truth_segment = "".join(translated_phrases)

            if not benchmark:
                print(
                    "\n[GROUND TRUTH SEGMENT (FIRST EVENTS AFTER PREDICTION TIME, "
                    f"UP TO {max_events} TOKENS)]\n"
                )
                print(ground_truth_segment)

        # If in benchmarking mode, compute metrics from the aggregated scores.
        if benchmark and labels_all:
            labels_np = np.array(labels_all, dtype=int)
            scores_np = np.array(scores_all, dtype=float)

            print("\n" + "=" * 80)
            print("BENCHMARKING RESULTS (death keyword frequency as risk score)")
            print("=" * 80)

            metrics = compute_baseline_metrics(labels_np, scores_np)
            print_results(metrics, split_name=split)

            # Optional: save curves & raw scores if output_dir is configured.
            output_dir = self.config.get("output_dir") or self.config.get("trajectory_output_dir")
            if output_dir:
                plot_all_curves(labels_np, scores_np, output_dir=output_dir + "/trajectory_benchmark")
                save_results(
                    {
                        "metrics": metrics,
                        "labels": labels_np.tolist(),
                        "scores": scores_np.tolist(),
                    },
                    output_dir=output_dir,
                    filename="trajectory_benchmark_results.json",
                )


def main():
    """
    Simple CLI entry point, mirroring the Pure LLM baseline style:

    python -m src.inference.trajectory_generator \\
        --config path/to/mimic_trajectory_config.yaml \\
        --split tuning \\
        --num_examples 5
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate future EHR trajectories from a pretrained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (same structure as Pure LLM but with model.checkpoint_dir).")
    parser.add_argument("--split", type=str, default="tuning", help="Dataset split to sample from (e.g. tuning, held_out).")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of patient trajectories to generate (overrides config if set).")
    parser.add_argument("--repetitions", type=int, default=None, help="Number of generations per trajectory (overrides config if set).")
    parser.add_argument("--benchmark", action="store_true", help="If set, run in benchmarking mode and compute ROC/AUPR.")

    args = parser.parse_args()

    config = load_baseline_config(args.config)
    runner = MIMICTrajectoryGenerationRunner(config)
    runner.run(
        split=args.split,
        num_examples=args.num_examples,
        repetitions=args.repetitions,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()

