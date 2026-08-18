#!/usr/bin/env python3
"""
Stratified ICU-mortality case selection + grounded post-hoc explanations.

Workflow
--------
1. Load ``test_results.json`` from a trained LLM classifier.
2. Pick 20 held-out examples (default):
     - 10 cases (label=1), 10 controls (label=0)
     - within each: 5 highest / 5 lowest probability of the *correct* class
       (covers confident-correct and hard / wrong fronts)
3. For each example, prompt the **base** LM (LoRA adapters disabled) with the
   record, the classifier's predicted probability, and a citation allow-list.
   A one-shot worked example fixes the output format so every claim carries a
   verbatim citation plus the actual value.
4. Score citations and value-grounding automatically; write JSON + markdown
   (full patient stream included) for manual review of the ~20 cases.

The true label is deliberately **not** in the prompt, so the explanation cannot
be reverse-engineered from the answer.

The LoRA adapters are not used for generation: they were trained for next-token
EHR modelling and classification, so with adapters on the model continues the
patient trajectory instead of explaining it.

Requires a GPU. Example::

    python -m src.analysis.prediction_explain \\
      --config_filepath src/configs/mimic/_resolved_isambard/explain-mistral-icu_mortality.yaml \\
      --results_json .../seed_42/results/test_results.json \\
      --out_dir analysis_out/explain-mistral-icu_mortality
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.training.config_utils import load_config
from src.training.training_utils import prepare_run_reproducibility, find_last_checkpoint
from src.data.dataset_factory import get_dataset


HEADER_SPLIT = " <HEADER_SPLIT> "
PHRASE_SPLIT_RE = re.compile(r";\s*")
# Citations may be wrapped in backticks, quotes, or left bare on a bullet line.
CITATION_PATTERNS = [
    re.compile(r"`([^`]+)`"),
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
]
BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
NUMBER_RE = re.compile(r"\d")
NUMERIC_PHRASE_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


# Synthetic patient for the one-shot example. Values are deliberately unlike
# anything in MIMIC cases so copied numbers are obvious during review.
FEWSHOT_HISTORY = (
    "Age; 68.0; Female; Race Black; Married; <HEADER_SPLIT> "
    "hospital admission, urgent; icu admission, coronary care unit (ccu); "
    "Creatinine; 2.8; Urea Nitrogen; 48.0; Potassium; 5.9; Bicarbonate; 15.0; "
    "Anion Gap; 22.0; Lactate; 6.1; Hemoglobin; 7.2; Platelet Count; 88.0; "
    "White Blood Cells; 21.4; "
    "Respiratory Ventilation, Greater than 96 Consecutive Hours; "
    "start NORepinephrine; start Vasopressin;"
)

FEWSHOT_ANSWER = """Drivers that increase the predicted risk:
Drivers that increase the predicted risk:
- Refractory Shock and Tissue Hypoperfusion: The patient requires escalating dual vasopressor support (`start NORepinephrine` with `start Vasopressin`), which corresponds with severe tissue hypoperfusion indicated by a critically high `Lactate; 6.1` (normal ~0.5-2.0 mmol/L).
- Acute Renal Failure: `Creatinine; 2.8` and `Urea Nitrogen; 48.0` are heavily elevated from normal baselines, indicating acute kidney injury likely secondary to the shock state, further complicated by arrhythmogenic hyperkalemia (`Potassium; 5.9`).
- High Anion-Gap Metabolic Acidosis: A `Bicarbonate; 15.0` paired with an `Anion Gap; 22.0` confirms severe metabolic acidosis, compounding the circulatory failure.
- Prolonged Respiratory Failure: The patient requires `Respiratory Ventilation, Greater than 96 Consecutive Hours`, highlighting severe, unresolved pulmonary compromise.
- Coagulopathy and Inflammation: A `Platelet Count; 88.0` suggests consumptive coagulopathy (normal ~150-450 K/uL), combined with significant leukocytosis (`White Blood Cells; 21.4`).

Drivers that decrease the predicted risk:
- There are no strong drivers decreasing risk, though the severe anemia (`Hemoglobin; 7.2`) is a highly treatable parameter via transfusion.

Assessment: Concurrent circulatory, renal, and respiratory failure with escalating vasopressor support paints a picture of multi-organ failure, strongly supporting the model's high predicted mortality."""


EXPLAIN_PROMPT = """You are a critical-care physician auditing an ICU mortality risk model.

The model reads a patient's structured ICU record and outputs the probability that the patient dies in the ICU. Explain what drives that number, using only the record.

Rules:
1. Clinical Synthesis: Do not just list isolated labs. Group related findings together to describe a clinical state (e.g., combining worsening kidney labs with vasopressor use, or pairing a diagnosis with a lab trend). 
2. Citation Format: You must support your synthesis by copying phrases verbatim from the record, wrapped in backticks (e.g., `Creatinine; 2.8`).
3. Strict Evidence & Values: ONLY cite clinical variables that have an explicit numerical value attached, or specific medications/events. If a lab lacks a value (e.g., `Asparate Aminotransferase (AST);`), ignore it completely.
4. Banned Items: Do NOT cite demographic factors (like race, marital status, or gender). Do not invent clinical definitions for single-letter codes (e.g., H, I, L) or unknown procedures.
5. Accuracy: Give the actual value of the labs you cite and ensure your clinical interpretation matches the value (e.g., do not call a high value "low").
6. Format: Group your points under "Drivers that increase the predicted risk" and "Drivers that decrease the predicted risk". Provide a MAXIMUM of 5 highly synthesized bullets total. 
7. Close with a one-sentence "Assessment" stating whether the record supports the model's number. No preamble.
--- Worked example ---

Record:
{fewshot_history}

Model output: probability of ICU death = 0.86 (cohort average 0.07)

Explanation:
{fewshot_answer}

--- Case to explain ---

Record:
{ehr_text}

Phrases you may cite (copy verbatim):
{allowed_citations}

Model output: probability of ICU death = {prob:.2f} (cohort average {prevalence:.2f})

Explanation:
"""


def _load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _split_phrases(text: str) -> Tuple[str, List[str]]:
    text = text.replace("<start>", "").replace("<end>", "").strip()
    if HEADER_SPLIT in text:
        header, body = text.split(HEADER_SPLIT, 1)
        header = header.strip()
        phrases = [p.strip() for p in PHRASE_SPLIT_RE.split(body) if p.strip()]
        return header, phrases
    phrases = [p.strip() for p in PHRASE_SPLIT_RE.split(text) if p.strip()]
    return "", phrases


def _normalize_phrase(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def _pair_names_with_values(phrases: Sequence[str]) -> List[str]:
    """
    Rejoin ``["Creatinine", "4.5"]`` into ``"Creatinine; 4.5"``.

    The record is semicolon-delimited, so a lab name and its value arrive as two
    phrases. Citing them separately is useless for grounding: a bare ``4.5``
    means nothing and a bare ``Creatinine`` carries no value. Labs recorded with
    no value are left bare, which is what prompt rule 5 asks the model to flag.
    """
    merged: List[str] = []
    i = 0
    while i < len(phrases):
        current = phrases[i]
        nxt = phrases[i + 1] if i + 1 < len(phrases) else None
        if (
            nxt is not None
            and NUMERIC_PHRASE_RE.match(nxt)
            and not NUMERIC_PHRASE_RE.match(current)
        ):
            merged.append(f"{current}; {nxt}")
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def _allowed_citations(text: str, max_phrases: int) -> List[str]:
    """
    Unique citable units from the record, as ``name; value`` where a value exists.

    Header fields (age / sex / race) are included so demographic claims can also
    be cited. When the record is longer than ``max_phrases`` the tail is kept,
    since recent events are usually the predictive ones.
    """
    header, phrases = _split_phrases(text)
    header_fields = [p.strip() for p in PHRASE_SPLIT_RE.split(header) if p.strip()]
    header_units = _pair_names_with_values(header_fields)
    body_units = _pair_names_with_values(phrases)
    # Cap the body only, so demographics stay citable on very long records.
    if len(body_units) > max_phrases:
        body_units = body_units[-max_phrases:]
    units = header_units + body_units
    if not units:
        return []
    seen = set()
    out: List[str] = []
    for p in units:
        key = _normalize_phrase(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _pick_stratified(
    labels: np.ndarray,
    probs: np.ndarray,
    n_per_cell: int,
) -> Dict[str, List[int]]:
    """
    2×2 grid on true label × confidence in the correct class.

    correct_class_prob = P(y) for positives, 1-P(y) for negatives.
    High = most confident in truth; low = least (errors / borderline).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    correct_prob = np.where(labels == 1, probs, 1.0 - probs)

    def take(mask: np.ndarray, high: bool) -> List[int]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        order = np.argsort(correct_prob[idx])
        if high:
            order = order[::-1]
        return [int(i) for i in idx[order[:n_per_cell]]]

    picks = {
        "case_high_correct_prob": take(labels == 1, high=True),
        "case_low_correct_prob": take(labels == 1, high=False),
        "control_high_correct_prob": take(labels == 0, high=True),
        "control_low_correct_prob": take(labels == 0, high=False),
    }
    # Drop accidental duplicates across high/low if the pool is tiny.
    used: set = set()
    deduped: Dict[str, List[int]] = {}
    for bucket, rows in picks.items():
        keep = []
        for r in rows:
            if r in used:
                continue
            used.add(r)
            keep.append(r)
        deduped[bucket] = keep
    return deduped


def _resolve_example_index(results: Dict[str, Any], row: int, dataset) -> int:
    if results.get("example_indices") and row < len(results["example_indices"]):
        return int(results["example_indices"][row])
    pid = str(results["patient_ids"][row])
    for i, ex in enumerate(dataset.examples):
        if str(ex["subject_id"]) == pid:
            return i
    raise KeyError(f"Could not resolve example for results row={row} patient_id={pid}")


def _format_prompt(
    tokenizer,
    ehr_text: str,
    prob: float,
    prevalence: float,
    allowed: Sequence[str],
    use_chat_template: bool,
) -> str:
    allowed_block = "\n".join(f"- {p}" for p in allowed) if allowed else "- (none extracted)"
    user_text = EXPLAIN_PROMPT.format(
        fewshot_history=FEWSHOT_HISTORY,
        fewshot_answer=FEWSHOT_ANSWER,
        ehr_text=ehr_text,
        allowed_citations=allowed_block,
        prob=prob,
        prevalence=prevalence,
    )
    if (
        use_chat_template
        and tokenizer is not None
        and getattr(tokenizer, "chat_template", None)
    ):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            print(f"  - WARNING: apply_chat_template failed ({exc}); using raw prompt")
    return user_text


def _truncate_for_context(
    tokenizer,
    prompt: str,
    max_input_tokens: int,
) -> str:
    """Keep the end of the prompt (task + recent EHR) if over budget."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_input_tokens:
        return prompt
    truncated = ids[-max_input_tokens:]
    print(
        f"  - Truncated prompt {len(ids)} → {len(truncated)} tokens "
        "(kept tail: citations + task + recent history)"
    )
    return tokenizer.decode(truncated, skip_special_tokens=False)


def _disable_adapters(model) -> bool:
    """Strip LoRA deltas so generation runs on the underlying base weights."""
    if hasattr(model, "disable_adapter_layers"):
        try:
            model.disable_adapter_layers()
            print("  - LoRA adapters disabled; generating from base weights")
            return True
        except Exception as exc:
            print(f"  - WARNING: disable_adapter_layers failed ({exc})")
    print(
        "  - WARNING: could not disable LoRA adapters; generations may not be "
        "pure base weights"
    )
    return False


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5) if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _extract_cited_spans(text: str) -> List[str]:
    spans: List[str] = []
    for pat in CITATION_PATTERNS:
        spans.extend(m.group(1).strip() for m in pat.finditer(text) if m.group(1).strip())
    return spans


def _citation_stats(generation: str, allowed: Sequence[str]) -> Dict[str, Any]:
    allowed_norm = {_normalize_phrase(a): a for a in allowed}
    cited = _extract_cited_spans(generation)
    attested = []
    invented = []
    for c in cited:
        key = _normalize_phrase(c)
        # Exact allow-list hit, or cited span is a substring of an allowed phrase
        # / allowed phrase is a substring of the citation (minor paraphrase slack).
        hit = key in allowed_norm
        if not hit:
            hit = any(
                key in _normalize_phrase(a) or _normalize_phrase(a) in key
                for a in allowed
            )
        if hit:
            attested.append(c)
        else:
            invented.append(c)
    return {
        "n_citations": len(cited),
        "n_attested": len(attested),
        "n_invented": len(invented),
        "attested": attested,
        "invented": invented,
        "all_citations_attested": len(invented) == 0 and len(cited) > 0,
    }


def _grounding_stats(generation: str) -> Dict[str, Any]:
    """
    Cheap proxy for "is this actually grounded" — a claim without a number or a
    citation is the vague kind we are trying to prompt away.
    """
    bullets = [ln for ln in generation.splitlines() if BULLET_RE.match(ln)]
    with_value = [ln for ln in bullets if NUMBER_RE.search(ln)]
    with_citation = [ln for ln in bullets if "`" in ln]
    n = len(bullets)
    return {
        "n_bullets": n,
        "n_bullets_with_value": len(with_value),
        "n_bullets_with_citation": len(with_citation),
        "frac_bullets_with_value": (len(with_value) / n) if n else 0.0,
        "frac_bullets_with_citation": (len(with_citation) / n) if n else 0.0,
        "ungrounded_bullets": [
            ln.strip() for ln in bullets if "`" not in ln or not NUMBER_RE.search(ln)
        ],
    }


def _resolve_paths(
    cfg: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, Optional[str]]:
    """
    Resolve the results dump and, for provenance only, a classifier checkpoint.

    The checkpoint is optional: probabilities come from ``test_results.json`` and
    generation uses the base LM, so no classifier weights need loading.
    """
    results_json = args.results_json
    ckpt = args.checkpoint
    out_dir = cfg["training"]["output_dir"]

    if not results_json:
        candidate = os.path.join(out_dir, "results", "test_results.json")
        if os.path.isfile(candidate):
            results_json = candidate
        else:
            raise FileNotFoundError(
                "Pass --results_json or ensure "
                f"{candidate} exists (run classifier eval first)."
            )

    if not ckpt:
        final_model = os.path.join(out_dir, "final_model")
        if os.path.isdir(final_model):
            ckpt = final_model
        else:
            ckpt = find_last_checkpoint(out_dir)
    return results_json, ckpt


def _prepare_gen_model(base_model, use_unsloth: bool):
    model = base_model
    model.eval()
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
        except Exception as exc:
            print(f"  - WARNING: FastLanguageModel.for_inference failed ({exc})")
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    return model


def run(args: argparse.Namespace) -> None:
    cfg = load_config(os.path.abspath(args.config_filepath))
    seed, _ = prepare_run_reproducibility(
        cfg, cli_seed=args.seed, use_seed_subdir=True
    )

    results_json, ckpt = _resolve_paths(cfg, args)
    results = _load_results(results_json)
    labels = np.asarray(results["labels"])
    probs = np.asarray(results["probs"])
    prevalence = float(labels.mean()) if len(labels) else 0.0
    picks = _pick_stratified(labels, probs, args.n_per_cell)

    data_cfg = cfg["data"]
    dataset = get_dataset(data_cfg.get("dataset", "MIMIC"), data_cfg, args.split)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Config seed/output_dir: {seed} / {cfg['training']['output_dir']}")
    print(f"Results : {results_json}")
    print(f"Checkpoint (provenance only): {ckpt}")
    print(f"Cohort prevalence: {prevalence:.4f}")
    print(f"Picks: { {k: len(v) for k, v in picks.items()} }")

    case_cards: List[Dict[str, Any]] = []
    for bucket, rows in picks.items():
        for row in rows:
            ex_i = _resolve_example_index(results, row, dataset)
            item = dataset[ex_i]
            text = item["text"]
            prob = float(probs[row])
            label = int(labels[row])
            pred = int(prob >= 0.5)
            allowed = _allowed_citations(text, max_phrases=args.max_citation_phrases)
            case_cards.append(
                {
                    "bucket": bucket,
                    "results_row": int(row),
                    "example_index": ex_i,
                    "subject_id": str(item.get("subject_id")),
                    "label": label,
                    "prob": prob,
                    "pred": pred,
                    "correct": pred == label,
                    "correct_class_prob": float(prob if label == 1 else 1.0 - prob),
                    "allowed_citations": allowed,
                    "text_chars": len(text),
                    "text": text,
                }
            )

    selection_path = os.path.join(args.out_dir, "selected_cases.json")
    with open(selection_path, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config_filepath),
                "results_json": os.path.abspath(results_json),
                "checkpoint": os.path.abspath(ckpt) if ckpt else None,
                "n_per_cell": args.n_per_cell,
                "prevalence": prevalence,
                "picks": picks,
                "cases": [
                    {k: v for k, v in c.items() if k != "text"} for c in case_cards
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote selection → {selection_path}")

    if args.select_only:
        print("--select_only set; skipping model load / generation.")
        _write_review_stub(args.out_dir, case_cards)
        return

    # Heavy imports (Unsloth) only when generating.
    from src.training.classification_trainer import EHRClassificationTrainer

    print("Loading base LM for explanation generation...")
    trainer = EHRClassificationTrainer(cfg)
    trainer.load_pretrained_model()
    gen_model = _prepare_gen_model(
        trainer.base_model,
        use_unsloth=bool(cfg.get("model", {}).get("use_unsloth", True)),
    )
    adapters_disabled = _disable_adapters(gen_model)
    tokenizer = trainer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_length = int(cfg["model"].get("max_length", 8192))
    max_input_tokens = max(1024, max_length - args.max_new_tokens - 8)
    use_chat = bool(
        cfg.get("model", {}).get("use_chat_template", False)
        or getattr(tokenizer, "chat_template", None)
    )
    # Base Mistral (non-instruct) often still has a template; allow CLI override.
    if args.force_chat_template is not None:
        use_chat = bool(args.force_chat_template)
    print(f"  - use_chat_template={use_chat}")

    for i, c in enumerate(case_cards):
        print(
            f"[{i + 1}/{len(case_cards)}] {c['bucket']} "
            f"subject={c['subject_id']} label={c['label']} p={c['prob']:.3f}"
        )
        prompt = _format_prompt(
            tokenizer,
            c["text"],
            c["prob"],
            prevalence,
            c["allowed_citations"],
            use_chat_template=use_chat,
        )
        prompt = _truncate_for_context(tokenizer, prompt, max_input_tokens)
        c["prompt_chars"] = len(prompt)
        text_out = _generate(
            gen_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        citation_check = _citation_stats(text_out, c["allowed_citations"])
        grounding = _grounding_stats(text_out)
        c["generation"] = {
            "text": text_out,
            "citation_check": citation_check,
            "grounding": grounding,
        }
        print(
            f"    {len(text_out)} chars, "
            f"citations={citation_check['n_citations']} "
            f"attested={citation_check['n_attested']} "
            f"invented={citation_check['n_invented']}, "
            f"bullets={grounding['n_bullets']} "
            f"with_value={grounding['n_bullets_with_value']}"
        )

    out_json = os.path.join(args.out_dir, "explanations.json")
    serialisable = [{k: v for k, v in c.items() if k != "text"} for c in case_cards]
    with open(out_json, "w") as f:
        json.dump(
            {
                "checkpoint": os.path.abspath(ckpt) if ckpt else None,
                "results_json": os.path.abspath(results_json),
                "prevalence": prevalence,
                "adapters_disabled": adapters_disabled,
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                },
                "cases": serialisable,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    md_path = os.path.join(args.out_dir, "explanations.md")
    _write_markdown(md_path, case_cards, prevalence)
    print(f"Wrote {md_path}")
    _write_review_stub(args.out_dir, case_cards)
    print(f"Done. Review {md_path} and review_rubric.md")


def _write_markdown(
    path: str, cases: List[Dict[str, Any]], prevalence: float
) -> None:
    with open(path, "w") as f:
        f.write("# Prediction explanations (base LM, grounded prompt)\n\n")
        f.write(
            "The prompt gives the classifier's predicted probability plus a "
            "citation allow-list, and a one-shot example fixes the format. The "
            "true label is **not** shown to the model. Generation uses the base "
            "weights with LoRA adapters disabled.\n\n"
            f"Cohort prevalence: {prevalence:.4f}\n\n"
        )
        for c in cases:
            f.write(
                f"## [{c['bucket']}] subject={c['subject_id']} "
                f"label={c['label']} pred={c['pred']} p={c['prob']:.3f} "
                f"correct={c['correct']}\n\n"
            )
            gen = c.get("generation") or {}
            stats = gen.get("citation_check") or {}
            grounding = gen.get("grounding") or {}
            f.write(
                f"### Explanation "
                f"(citations attested={stats.get('n_attested', 0)}/"
                f"{stats.get('n_citations', 0)}, "
                f"invented={stats.get('n_invented', 0)}; "
                f"bullets with a value="
                f"{grounding.get('n_bullets_with_value', 0)}/"
                f"{grounding.get('n_bullets', 0)})\n\n"
            )
            f.write(gen.get("text") or "_(empty)_")
            f.write("\n\n")
            if stats.get("invented"):
                f.write(
                    "**Invented citations:** "
                    + ", ".join(f"`{x}`" for x in stats["invented"])
                    + "\n\n"
                )
            if grounding.get("ungrounded_bullets"):
                f.write("**Bullets missing a citation or a value:**\n\n")
                for ln in grounding["ungrounded_bullets"]:
                    f.write(f"- {ln}\n")
                f.write("\n")
            f.write(
                f"### Full patient stream ({c['text_chars']} chars, "
                f"{len(c['allowed_citations'])} citable phrases)\n\n```\n"
            )
            f.write(c["text"])
            f.write("\n```\n\n")


def _write_review_stub(out_dir: str, cases: List[Dict[str, Any]]) -> None:
    path = os.path.join(out_dir, "review_rubric.md")
    with open(path, "w") as f:
        f.write("# Manual review rubric\n\n")
        f.write(
            "Score each explanation 0-2 on: relevance to the predicted risk, "
            "temporal/trend awareness, value grounding (does every claim quote a "
            "real number or event), and absence of invented content. The "
            "citation and value counts in `explanations.md` are automatic; the "
            "columns below are for your judgement.\n\n"
        )
        f.write(
            "| subject | bucket | label | p | relevance | trends | grounding | "
            "no_invention | notes |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for c in cases:
            f.write(
                f"| {c['subject_id']} | {c['bucket']} | {c['label']} | "
                f"{c['prob']:.3f} |  |  |  |  |  |\n"
            )
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description="Stratified case pick + grounded post-hoc explanations"
    )
    p.add_argument("--config_filepath", required=True)
    p.add_argument(
        "--results_json",
        default=None,
        help="test_results.json (default: <output_dir>/results/test_results.json)",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Recorded for provenance only; generation uses the base LM",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="held_out")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--n_per_cell",
        type=int,
        default=5,
        help="Examples per cell of the 2×2 (label × high/low correct-class prob). "
        "Default 5 → 20 total.",
    )
    p.add_argument(
        "--select_only",
        action="store_true",
        help="Only write selected_cases.json (no GPU generation)",
    )
    p.add_argument("--max_citation_phrases", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument(
        "--force_chat_template",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=None,
        help="true/false to force chat template on/off",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
