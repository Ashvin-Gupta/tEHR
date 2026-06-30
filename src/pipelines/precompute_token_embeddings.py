"""
Precompute token embeddings for the text-embedding transformer baseline.

For each unique token in the MIMIC vocabulary this script:
  1. Translates the token ID → human-readable string via MIMICTokenTranslator.
  2. Embeds the string with a frozen language model (e.g. Qwen3-Embedding-0.6B
     or Bio_ClinicalBERT).
  3. Saves a .pt file containing:
        {
            "embeddings":       FloatTensor (N_tokens, D),
            "token_id_to_row":  dict[int, int],   # token_id → row index
            "dim":              int,               # embedding dimension D
            "model_name":       str,
        }

Usage:
    python -m src.pipelines.precompute_token_embeddings \\
        --vocab_filepath  "/home/ag619/MIMIC-IV tokensiation/vocab.csv" \\
        --lookup_dir      "/home/ag619/MIMIC_data/hosp" \\
        --model_name      "Qwen/Qwen3-Embedding-0.6B" \\
        --output_path     "/home/ag619/experiments/Results/mimic-baselines/text_embedding_transformer/token_embeddings.pt" \\
        [--batch_size 256] \\
        [--max_length 64] \\
        [--device cuda]
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.data.mimic_translator import MIMICTokenTranslator


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over token hidden states."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_strings(
    strings: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    max_length: int = 64,
) -> torch.Tensor:
    """
    Embed a list of strings and return a (N, D) float32 tensor.

    Uses masked mean pooling over the last hidden state.
    For models that expose a `sentence_embedding` output (e.g. some Qwen
    variants), that is used directly instead.
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(strings), batch_size), desc="Embedding tokens"):
            batch = strings[i: i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**enc)

            # Some embedding models expose `.sentence_embedding`
            if hasattr(outputs, "sentence_embedding"):
                emb = outputs.sentence_embedding
            else:
                emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])

            # L2-normalise for cosine-friendly embeddings
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embeddings.append(emb.cpu().float())

    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Token translation
# ---------------------------------------------------------------------------

def _translate_token(token_str: str, translator: MIMICTokenTranslator) -> str:
    """
    Translate a raw MIMIC token string to a human-readable string for embedding.

    Falls back to the raw string if translation returns an empty or
    error result.
    """
    if not isinstance(token_str, str) or not token_str.strip():
        return "<unknown>"
    translated = translator.translate(token_str)
    # Strip trailing semicolons / whitespace added by the translator
    cleaned = translated.strip().rstrip(";").strip()
    if not cleaned or cleaned == "---ERROR_TRANSLATING---":
        return token_str  # fall back to the raw string
    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load vocab
    # -----------------------------------------------------------------------
    print(f"\nLoading vocabulary from: {args.vocab_filepath}")
    vocab_df = pd.read_csv(args.vocab_filepath, dtype={"str": str})
    # vocab_df columns: 'token' (int ID), 'str' (raw token string)
    token_ids: list[int] = vocab_df["token"].tolist()
    token_strings: list[str] = vocab_df["str"].tolist()
    print(f"  Vocabulary size: {len(token_ids):,}")

    # -----------------------------------------------------------------------
    # 2. Translate tokens to human-readable strings
    # -----------------------------------------------------------------------
    print(f"\nLoading MIMIC translator from: {args.lookup_dir}")
    translator = MIMICTokenTranslator.from_csv_files(
        labitems_file=os.path.join(args.lookup_dir, "d_labitems.csv.gz"),
        diagnoses_file=os.path.join(args.lookup_dir, "d_icd_diagnoses.csv.gz"),
        procedures_file=os.path.join(args.lookup_dir, "d_icd_procedures.csv.gz"),
    )

    print("Translating tokens to human-readable strings...")
    readable_strings: list[str] = [
        _translate_token(s, translator) for s in tqdm(token_strings, desc="Translating")
    ]

    # Brief sample
    for tok_id, raw, readable in zip(token_ids[:5], token_strings[:5], readable_strings[:5]):
        print(f"  [{tok_id}] '{raw}' → '{readable}'")

    # -----------------------------------------------------------------------
    # 3. Load embedding model
    # -----------------------------------------------------------------------
    print(f"\nLoading embedding model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    # Determine embedding dimension with a dummy forward pass
    with torch.no_grad():
        dummy_enc = tokenizer(["test"], return_tensors="pt").to(device)
        dummy_out = model(**dummy_enc)
        if hasattr(dummy_out, "sentence_embedding"):
            embed_dim = dummy_out.sentence_embedding.shape[-1]
        else:
            embed_dim = dummy_out.last_hidden_state.shape[-1]
    print(f"  Embedding dimension: {embed_dim}")

    # -----------------------------------------------------------------------
    # 4. Embed all readable strings
    # -----------------------------------------------------------------------
    print(f"\nEmbedding {len(readable_strings):,} token strings...")
    embeddings = embed_strings(
        readable_strings,
        model,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"  Embeddings shape: {tuple(embeddings.shape)}")

    # -----------------------------------------------------------------------
    # 5. Build token_id → row mapping and save
    # -----------------------------------------------------------------------
    token_id_to_row: dict[int, int] = {
        int(tok_id): row_idx for row_idx, tok_id in enumerate(token_ids)
    }

    save_dict = {
        "embeddings": embeddings,       # (N_tokens, D) float32
        "token_id_to_row": token_id_to_row,
        "dim": embed_dim,
        "model_name": args.model_name,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(save_dict, args.output_path)
    print(f"\nSaved embeddings to: {args.output_path}")
    print(f"  File size: {os.path.getsize(args.output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute MIMIC token embeddings")
    parser.add_argument(
        "--vocab_filepath",
        type=str,
        required=True,
        help="Path to vocab.csv (columns: token, str)",
    )
    parser.add_argument(
        "--lookup_dir",
        type=str,
        required=True,
        help="Directory containing d_labitems.csv.gz, d_icd_diagnoses.csv.gz, d_icd_procedures.csv.gz",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Max token length for the embedder (token strings are short)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'",
    )
    args = parser.parse_args()
    main(args)
