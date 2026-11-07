"""
Precompute phrase embeddings for the text-embedding transformer baseline.

For each unique natural-language *phrase* that appears in patient trajectories
this script:
  1. Builds phrases the same way as the LLM path (``tokens_to_phrases``),
     fusing measurable concept + value (+ optional unit) into one string.
  2. Embeds each phrase with a frozen language model using **last-token
     pooling** (required for Qwen3-Embedding) + L2 normalisation.
  3. Saves a .pt file containing:
        {
            "embeddings":      FloatTensor (N_phrases, D),
            "phrase_to_row":   dict[str, int],   # phrase → row index
            "phrases":         list[str],        # row → phrase
            "dim":             int,
            "model_name":      str,
            "pooling":         "last_token",
            "unit":            "phrase",
        }

Usage:
    python -m src.pipelines.precompute_token_embeddings \\
        --vocab_filepath  ".../vocab.csv" \\
        --lookup_dir      ".../hosp" \\
        --data_dir        ".../MIMIC-IV_tokenisation" \\
        --model_name      "Qwen/Qwen3-Embedding-0.6B" \\
        --output_path     ".../phrase_embeddings.pt" \\
        [--batch_size 256] \\
        [--max_length 128] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd
import torch
from tqdm import tqdm

from src.data.base_dataset import tokens_to_phrases
from src.data.base_token_translator import BaseTokenTranslator
from src.data.cprd_translator import CPRDTokenTranslator
from src.data.ehrshot_translator import EHRSHOTTokenTranslator
from src.data.mimic_translator import MIMICTokenTranslator


# ---------------------------------------------------------------------------
# Embedding utilities (Qwen3-Embedding: last-token pool + L2 normalise)
# ---------------------------------------------------------------------------

def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Pool at the last non-pad token (Qwen3-Embedding convention).

    With left padding every row's last index is valid; with right padding we
    index by ``attention_mask.sum(dim=1) - 1``.
    """
    left_padding = bool(torch.all(attention_mask[:, -1] == 1).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1).long() - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _hidden_states(model, enc: dict, kind: str) -> torch.Tensor:
    """Final token states. Gemma 3 is the multimodal class; text-only skips vision."""
    if kind == "gemma3":
        token_type_ids = torch.zeros_like(enc["input_ids"])
        outputs = model.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=token_type_ids,
            use_cache=False,
        )
        return outputs.last_hidden_state
    outputs = model(**enc)
    return outputs.last_hidden_state


def load_embedder(model_name: str, device: torch.device):
    """Load a frozen phrase embedder. Gemma 3 stays in bfloat16."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    architectures = list(getattr(config, "architectures", None) or [])
    gemma3 = (
        getattr(config, "model_type", "") == "gemma3"
        or "Gemma3ForConditionalGeneration" in architectures
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if gemma3:
        from transformers import AutoModelForImageTextToText

        tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        kind = "gemma3"
    else:
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device)
        kind = "standard"
    model.eval()
    return model, tokenizer, kind


def embed_strings(
    strings: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 128,
    kind: str = "standard",
) -> torch.Tensor:
    """Embed strings with last-token pooling; return (N, D) float32 L2-normalised."""
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(strings), batch_size), desc="Embedding phrases"):
            batch = strings[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}

            hidden = _hidden_states(model, enc, kind)
            emb = last_token_pool(hidden, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Phrase collection
# ---------------------------------------------------------------------------

def _iter_patient_token_sequences(data_dir: str, splits: Sequence[str]) -> Iterable[List[int]]:
    """Yield raw token-id sequences from pickle shards under data_dir/<split>/."""
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  Warning: split dir missing, skipping: {split_dir}")
            continue
        pkl_files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".pkl")
        )
        for path in tqdm(pkl_files, desc=f"Scanning {split}"):
            with open(path, "rb") as f:
                records = pickle.load(f)
            for record in records:
                tokens = record.get("tokens")
                if tokens is None:
                    continue
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                yield list(tokens)


def collect_unique_phrases(
    data_dir: str,
    id_to_token: Dict[int, str],
    translator: BaseTokenTranslator,
    splits: Sequence[str] = ("train", "tuning", "held_out"),
) -> List[str]:
    """
    Collect unique NL phrases from patient trajectories.

    Uses the same pairing rules as the LLM narrative path, without HEADER_SPLIT
    markers (those are not semantic events).
    """
    unique: Set[str] = set()
    n_sequences = 0
    for token_ids in _iter_patient_token_sequences(data_dir, splits):
        n_sequences += 1
        string_codes = [id_to_token.get(int(tid), "") for tid in token_ids]
        for phrase in tokens_to_phrases(
            string_codes, translator, include_header_split=False
        ):
            cleaned = phrase.strip()
            if cleaned:
                unique.add(cleaned)

    phrases = sorted(unique)
    print(
        f"  Collected {len(phrases):,} unique phrases "
        f"from {n_sequences:,} patient sequences"
    )
    return phrases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading vocabulary from: {args.vocab_filepath}")
    vocab_df = pd.read_csv(args.vocab_filepath, dtype={"str": str})
    id_to_token: Dict[int, str] = {
        int(tok_id): str(s) for tok_id, s in zip(vocab_df["token"], vocab_df["str"])
    }
    print(f"  Vocabulary size: {len(id_to_token):,}")

    dataset = args.dataset.lower()
    if dataset == "mimic":
        if not args.lookup_dir:
            raise SystemExit("--lookup_dir is required for dataset=mimic")
        print(f"\nLoading MIMIC translator from: {args.lookup_dir}")
        translator = MIMICTokenTranslator.from_csv_files(
            labitems_file=os.path.join(args.lookup_dir, "d_labitems.csv.gz"),
            diagnoses_file=os.path.join(args.lookup_dir, "d_icd_diagnoses.csv.gz"),
            procedures_file=os.path.join(args.lookup_dir, "d_icd_procedures.csv.gz"),
        )
    elif dataset == "cprd":
        missing = [
            name for name in (
                "medical_lookup_filepath",
                "lab_lookup_filepath",
                "region_lookup_filepath",
                "time_lookup_filepath",
            )
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(
                "CPRD phrase embeddings require: "
                + ", ".join("--" + m for m in missing)
            )
        print("\nLoading CPRD translator from lookup CSVs")
        translator = CPRDTokenTranslator.from_csv_files(
            medical_lookup_file=args.medical_lookup_filepath,
            lab_lookup_file=args.lab_lookup_filepath,
            region_lookup_file=args.region_lookup_filepath,
            time_lookup_file=args.time_lookup_filepath,
        )
    elif dataset == "ehrshot":
        if not args.concept_csv:
            raise SystemExit("--concept_csv is required for dataset=ehrshot")
        print(f"\nLoading EHRSHOT translator from: {args.concept_csv}")
        translator = EHRSHOTTokenTranslator.from_concept_csv(args.concept_csv)
    else:
        raise SystemExit(
            f"Unknown dataset '{args.dataset}'. Use mimic, cprd, or ehrshot."
        )

    print(f"\nCollecting unique phrases from: {args.data_dir}")
    phrases = collect_unique_phrases(args.data_dir, id_to_token, translator)
    if not phrases:
        raise RuntimeError(
            f"No phrases collected from {args.data_dir}. "
            "Check --data_dir points at tokenised splits."
        )

    for p in phrases[:5]:
        print(f"  sample phrase: {p!r}")

    print(f"\nLoading embedding model: {args.model_name}")
    model, tokenizer, kind = load_embedder(args.model_name, device)
    if kind == "gemma3" and args.batch_size > 32:
        print(f"  Gemma 3 embedder: lowering batch size {args.batch_size} -> 32")
        args.batch_size = 32

    with torch.no_grad():
        dummy_enc = tokenizer(
            ["test"],
            padding=True,
            return_tensors="pt",
        )
        dummy_enc = {key: value.to(device) for key, value in dummy_enc.items()}
        dummy_hidden = _hidden_states(model, dummy_enc, kind)
        embed_dim = int(last_token_pool(
            dummy_hidden, dummy_enc["attention_mask"]
        ).shape[-1])
    print(f"  Embedding dimension: {embed_dim} (pooling=last_token, kind={kind})")

    print(f"\nEmbedding {len(phrases):,} unique phrases...")
    embeddings = embed_strings(
        phrases,
        model,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        kind=kind,
    )
    print(f"  Embeddings shape: {tuple(embeddings.shape)}")
    norms = embeddings.norm(dim=-1)
    print(
        f"  L2 norms: mean={float(norms.mean()):.4f} "
        f"min={float(norms.min()):.4f} max={float(norms.max()):.4f}"
    )

    phrase_to_row = {p: i for i, p in enumerate(phrases)}
    save_dict = {
        "embeddings": embeddings,
        "phrase_to_row": phrase_to_row,
        "phrases": phrases,
        "dim": embed_dim,
        "model_name": args.model_name,
        "pooling": "last_token",
        "unit": "phrase",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    partial_path = args.output_path + ".partial"
    torch.save(save_dict, partial_path)
    os.replace(partial_path, args.output_path)
    print(f"\nSaved phrase embeddings to: {args.output_path}")
    print(f"  File size: {os.path.getsize(args.output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute phrase embeddings (last-token pool) for MIMIC, CPRD, or EHRSHOT"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic",
        choices=("mimic", "cprd", "ehrshot"),
        help="Which token translator to use",
    )
    parser.add_argument(
        "--vocab_filepath",
        type=str,
        required=True,
        help="Path to vocab.csv (columns: token, str)",
    )
    parser.add_argument(
        "--lookup_dir",
        type=str,
        default=None,
        help="MIMIC lookup dir (d_labitems / d_icd_diagnoses / d_icd_procedures)",
    )
    parser.add_argument("--medical_lookup_filepath", type=str, default=None)
    parser.add_argument("--lab_lookup_filepath", type=str, default=None)
    parser.add_argument("--region_lookup_filepath", type=str, default=None)
    parser.add_argument("--time_lookup_filepath", type=str, default=None)
    parser.add_argument(
        "--concept_csv",
        type=str,
        default=None,
        help="EHRSHOT Athena concept.csv (vocabulary_id / concept_code / concept_name)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Tokenised data root with train/tuning/held_out pickle shards",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model name for the embedder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .pt file path",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max tokenizer length for phrases (lab+value+unit can be longer than atoms)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
