#!/usr/bin/env python3
"""
Qualitative case dump + phrase-occlusion analysis (paper analysis exp 5).

1. Selects interesting held-out cases (high-confidence correct / incorrect).
2. Optionally runs leave-one-phrase-out occlusion to rank clinically salient phrases.

Requires a trained LLM classifier checkpoint (``final_model`` or ``checkpoint-*``).

Examples:
  # Dump cases only (still loads model to re-score baseline; use --dump_only to skip GPU)
  python -m src.analysis.qualitative_occlusion \\
    --config_filepath src/configs/mimic/_resolved_isambard/classifier-mistral-icu_mortality.yaml \\
    --results_json .../seed_42/results/test_results.json \\
    --checkpoint .../seed_42/final_model \\
    --out_dir analysis_out/qual_icu \\
    --n_cases 5

  # Cases + occlusion (GPU)
  python -m src.analysis.qualitative_occlusion \\
    --config_filepath .../classifier-mistral-icu_mortality.yaml \\
    --results_json .../test_results.json \\
    --checkpoint .../final_model \\
    --out_dir analysis_out/qual_icu \\
    --occlude --max_phrases 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.training.config_utils import load_config
from src.training.training_utils import prepare_run_reproducibility, find_last_checkpoint
from src.training.classification_trainer import EHRClassificationTrainer
from src.data.dataset_factory import get_dataset
from src.data.classification_collator import ClassificationCollator


HEADER_SPLIT = " <HEADER_SPLIT> "
PHRASE_SPLIT_RE = re.compile(r";\s*")


def _load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _split_phrases(text: str) -> Tuple[str, List[str]]:
    """
    Split narrative into header (kept intact) + body phrases.

    Returns (header_with_split_or_empty, body_phrases).
    """
    text = text.replace("<start>", "").replace("<end>", "").strip()
    if HEADER_SPLIT in text:
        header, body = text.split(HEADER_SPLIT, 1)
        header = header.strip()
        phrases = [p.strip() for p in PHRASE_SPLIT_RE.split(body) if p.strip()]
        return header, phrases
    phrases = [p.strip() for p in PHRASE_SPLIT_RE.split(text) if p.strip()]
    return "", phrases


def _rebuild_text(header: str, phrases: List[str]) -> str:
    body = "; ".join(phrases)
    if body and not body.endswith(";"):
        body = body + ";"
    if header:
        return f"{header}{HEADER_SPLIT}{body}"
    return body


def _logits_to_pos_prob(logits: torch.Tensor) -> float:
    if logits.ndim == 2 and logits.shape[1] == 1:
        return float(torch.sigmoid(logits.squeeze()).item())
    probs = torch.softmax(logits, dim=-1)
    if probs.shape[-1] >= 2:
        return float(probs[0, 1].item())
    return float(probs.reshape(-1)[0].item())


@torch.inference_mode()
def _score_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> List[float]:
    collator = ClassificationCollator(tokenizer, max_length=max_length, truncation=True)
    batch_items = [{"text": t, "label": 0} for t in texts]
    batch = collator(batch_items)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]
    probs = []
    for i in range(logits.shape[0]):
        probs.append(_logits_to_pos_prob(logits[i : i + 1]))
    return probs


def _pick_cases(
    labels: np.ndarray,
    probs: np.ndarray,
    n_cases: int,
) -> Dict[str, List[int]]:
    """Pick interesting row indices from a results dump."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    correct = preds == labels
    incorrect = ~correct

    def top_k(mask: np.ndarray, score: np.ndarray, k: int, descending: bool = True) -> List[int]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        order = np.argsort(score[idx])
        if descending:
            order = order[::-1]
        return [int(i) for i in idx[order[:k]]]

    # High-confidence correct positives / negatives
    tp = correct & (labels == 1)
    tn = correct & (labels == 0)
    fp = incorrect & (preds == 1)
    fn = incorrect & (preds == 0)

    return {
        "correct_pos_high": top_k(tp, probs, n_cases, descending=True),
        "correct_neg_high": top_k(tn, 1.0 - probs, n_cases, descending=True),
        "false_pos_high": top_k(fp, probs, n_cases, descending=True),
        "false_neg_high": top_k(fn, 1.0 - probs, n_cases, descending=True),  # low prob on positives
    }


def _resolve_example_index(
    results: Dict[str, Any],
    row: int,
    dataset,
) -> int:
    if results.get("example_indices") and row < len(results["example_indices"]):
        return int(results["example_indices"][row])
    # Fall back: match subject_id greedily
    pid = str(results["patient_ids"][row])
    for i, ex in enumerate(dataset.examples):
        if str(ex["subject_id"]) == pid:
            return i
    raise KeyError(f"Could not resolve example for results row={row} patient_id={pid}")


def _occlude_case(
    model,
    tokenizer,
    text: str,
    max_length: int,
    device: torch.device,
    max_phrases: int,
    batch_size: int = 4,
) -> Dict[str, Any]:
    header, phrases = _split_phrases(text)
    if not phrases:
        base = _score_texts(model, tokenizer, [text], max_length, device)[0]
        return {"baseline_prob": base, "phrase_deltas": []}

    base_text = _rebuild_text(header, phrases)
    baseline = _score_texts(model, tokenizer, [base_text], max_length, device)[0]

    # Cap phrases (prefer trailing / recent events — usually more predictive)
    if len(phrases) > max_phrases:
        start = len(phrases) - max_phrases
        occlude_indices = list(range(start, len(phrases)))
    else:
        occlude_indices = list(range(len(phrases)))

    deltas = []
    # Score in batches
    for bstart in range(0, len(occlude_indices), batch_size):
        chunk = occlude_indices[bstart : bstart + batch_size]
        texts = []
        for j in chunk:
            kept = [p for i, p in enumerate(phrases) if i != j]
            texts.append(_rebuild_text(header, kept))
        probs = _score_texts(model, tokenizer, texts, max_length, device)
        for j, prob in zip(chunk, probs):
            deltas.append(
                {
                    "phrase_index": j,
                    "phrase": phrases[j][:300],
                    "prob_without": prob,
                    "delta": baseline - prob,  # >0 ⇒ phrase increased risk score
                }
            )

    deltas.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return {
        "baseline_prob": baseline,
        "n_phrases": len(phrases),
        "n_occluded": len(occlude_indices),
        "header": header[:400],
        "top_phrases": deltas[:20],
        "phrase_deltas": deltas,
    }


def run(args: argparse.Namespace) -> None:
    cfg = load_config(os.path.abspath(args.config_filepath))
    seed, _ = prepare_run_reproducibility(
        cfg, cli_seed=args.seed, use_seed_subdir=True
    )

    results = _load_results(args.results_json)
    labels = np.asarray(results["labels"])
    probs = np.asarray(results["probs"])
    picks = _pick_cases(labels, probs, args.n_cases)

    data_cfg = cfg["data"]
    dataset = get_dataset(data_cfg.get("dataset", "MIMIC"), data_cfg, args.split)

    # Resolve checkpoint
    ckpt = args.checkpoint
    if not ckpt:
        out_dir = cfg["training"]["output_dir"]
        final_model = os.path.join(out_dir, "final_model")
        if os.path.isdir(final_model):
            ckpt = final_model
        else:
            ckpt = find_last_checkpoint(out_dir)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint pointing at final_model "
            "or a Trainer checkpoint directory."
        )
    print(f"Using checkpoint: {ckpt}")
    print(f"Seed / output_dir: {seed} / {cfg['training']['output_dir']}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Build case cards (text dump always; occlusion optional)
    case_cards: List[Dict[str, Any]] = []
    for bucket, rows in picks.items():
        for row in rows:
            ex_i = _resolve_example_index(results, row, dataset)
            item = dataset[ex_i]
            text = item["text"]
            case_cards.append(
                {
                    "bucket": bucket,
                    "results_row": row,
                    "example_index": ex_i,
                    "subject_id": str(item.get("subject_id")),
                    "label": int(labels[row]),
                    "prob_saved": float(probs[row]),
                    "text_chars": len(text),
                    "text_preview": text[:1200],
                    "text": text if args.save_full_text else None,
                }
            )

    dump_path = os.path.join(args.out_dir, "cases.json")
    with open(dump_path, "w") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config_filepath),
                "results_json": os.path.abspath(args.results_json),
                "checkpoint": os.path.abspath(ckpt),
                "picks": {k: v for k, v in picks.items()},
                "cases": [
                    {k: v for k, v in c.items() if k != "text" or args.save_full_text}
                    for c in case_cards
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote case dump → {dump_path}")

    # Human-readable markdown summary
    md_path = os.path.join(args.out_dir, "cases.md")
    with open(md_path, "w") as f:
        f.write("# Qualitative cases\n\n")
        for c in case_cards:
            f.write(
                f"## [{c['bucket']}] subject={c['subject_id']} "
                f"label={c['label']} prob={c['prob_saved']:.3f}\n\n"
            )
            f.write("```\n")
            f.write(c["text_preview"])
            f.write("\n```\n\n")
    print(f"Wrote markdown → {md_path}")

    if args.dump_only:
        print("--dump_only set; skipping model load / occlusion.")
        return

    print("Loading classifier for scoring / occlusion...")
    trainer = EHRClassificationTrainer(cfg)
    trainer.load_pretrained_model()
    trainer.create_classifier()
    trainer.restore_weights_from_checkpoint(ckpt)
    model = trainer.classifier_model
    model.eval()
    device = next(model.parameters()).device
    max_length = int(cfg["model"].get("max_length", 8192))

    # Re-score baselines for dumped cases
    for c in case_cards:
        text = c.get("text") or dataset[c["example_index"]]["text"]
        c["prob_rescored"] = _score_texts(
            model, trainer.tokenizer, [text], max_length, device
        )[0]

    if args.occlude:
        print("Running phrase occlusion...")
        for i, c in enumerate(case_cards):
            text = c.get("text") or dataset[c["example_index"]]["text"]
            print(
                f"  [{i + 1}/{len(case_cards)}] {c['bucket']} "
                f"subject={c['subject_id']} ..."
            )
            c["occlusion"] = _occlude_case(
                model,
                trainer.tokenizer,
                text,
                max_length,
                device,
                max_phrases=args.max_phrases,
                batch_size=args.occlusion_batch_size,
            )

    out_json = os.path.join(args.out_dir, "qualitative_occlusion.json")
    serialisable = []
    for c in case_cards:
        row = {k: v for k, v in c.items() if k != "text"}
        serialisable.append(row)
    with open(out_json, "w") as f:
        json.dump(
            {
                "checkpoint": os.path.abspath(ckpt),
                "occlude": bool(args.occlude),
                "cases": serialisable,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json}")

    if args.occlude:
        md2 = os.path.join(args.out_dir, "occlusion_top_phrases.md")
        with open(md2, "w") as f:
            f.write("# Occlusion top phrases\n\n")
            f.write(
                "Δ = baseline_prob − prob_without_phrase. "
                "Positive Δ ⇒ phrase increased predicted risk.\n\n"
            )
            for c in case_cards:
                occ = c.get("occlusion") or {}
                f.write(
                    f"## [{c['bucket']}] subject={c['subject_id']} "
                    f"label={c['label']} baseline={occ.get('baseline_prob', float('nan')):.3f}\n\n"
                )
                for d in (occ.get("top_phrases") or [])[:10]:
                    f.write(
                        f"- Δ={d['delta']:+.4f}  without={d['prob_without']:.3f}  "
                        f"`{d['phrase']}`\n"
                    )
                f.write("\n")
        print(f"Wrote {md2}")


def main():
    p = argparse.ArgumentParser(description="Qualitative cases + phrase occlusion")
    p.add_argument("--config_filepath", required=True)
    p.add_argument("--results_json", required=True)
    p.add_argument("--checkpoint", default=None, help="final_model or checkpoint-* dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="held_out")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_cases", type=int, default=3, help="Cases per bucket")
    p.add_argument("--dump_only", action="store_true", help="No model load / occlusion")
    p.add_argument("--occlude", action="store_true", help="Run leave-one-phrase-out")
    p.add_argument("--max_phrases", type=int, default=40)
    p.add_argument("--occlusion_batch_size", type=int, default=2)
    p.add_argument(
        "--save_full_text",
        action="store_true",
        help="Include full trajectories in JSON (can be large)",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
