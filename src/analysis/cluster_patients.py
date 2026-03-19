"""
Patient clustering analysis using LLM + LoRA classifier representations.

This script:
    - Loads a trained classification model (base LLM + LoRA + head).
    - Extracts the patient-level representation that feeds into the classifier head.
    - Runs dimensionality reduction (UMAP by default).
    - Runs k-means clustering on the reduced (or original) space.
    - Links clusters back to raw EHR events for simple enrichment summaries.
"""

import argparse
import os
import json
import math
from typing import List, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipelines.evaluate_model import ModelEvaluator
from src.data.dataset_factory import get_dataset
from src.data.classification_collator import ClassificationCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster patients based on LLM + LoRA classifier representations."
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file (same as used for training).",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help=(
            "Path to saved model checkpoint directory "
            "(e.g., training output_dir/final_model)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="held_out",
        choices=["train", "tuning", "held_out"],
        help="Dataset split to use for clustering.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write embeddings, projections, clusters, and summaries.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional maximum number of patients to process (for quick tests).",
    )


    return parser.parse_args()


class ClusterCollator:
    """
    Wraps ClassificationCollator to also return patient IDs.

    - Uses the existing collator to build input_ids/attention_mask/labels.
    - Additionally collects 'subject_id' from each item (if present) and
      returns it as a list in the batch dict under 'patient_ids'.
    """

    def __init__(self, base_collator: ClassificationCollator):
        self.base_collator = base_collator

    def __call__(self, batch):
        patient_ids: List = []
        for idx, item in enumerate(batch):
            pid = item.get("subject_id", idx)
            patient_ids.append(pid)

        collated = self.base_collator(batch)
        collated["patient_ids"] = patient_ids
        return collated


def resolve_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    config_path: str,
    model_checkpoint: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, any, dict, dict]:
    """
    Reuse the evaluation pipeline's ModelEvaluator to load the trained model.

    Returns:
        model: LLMClassifier instance with base LLM + LoRA + classifier head.
        tokenizer: Associated tokenizer.
        data_config: Data section of the config.
        training_config: Training section of the config.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    evaluator = ModelEvaluator(config, model_checkpoint)
    evaluator.load_model()
    evaluator.wrap_with_classifier()

    model = evaluator.model.to(device)
    model.eval()

    return model, evaluator.tokenizer, evaluator.data_config, evaluator.training_config


def build_dataloader(
    data_config: dict,
    tokenizer,
    split: str,
    batch_size: int,
) -> Tuple[DataLoader, any]:
    """
    Build a DataLoader over the UnifiedEHRDataset for the given split.

    Returns:
        dataloader, dataset
    """
    data_config = {**data_config, "format": "text"}
    dataset_name = data_config.get("dataset", "CPRD")
    dataset = get_dataset(dataset_name, data_config, split)

    collator = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 12000),
    )
    cluster_collator = ClusterCollator(collator)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get("dataloader_num_workers", 4),
        collate_fn=cluster_collator,
    )

    return dataloader, dataset


def extract_patient_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_patients: int | None = None,
) -> Tuple[np.ndarray, List]:
    """
    Run the model over the dataset and collect patient-level representations.

    Returns:
        embeddings: (N, hidden_dim) numpy array.
        patient_ids: list of patient identifiers aligned with embeddings.
    """
    all_embeddings: List[np.ndarray] = []
    all_patient_ids: List = []

    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            patient_ids = batch["patient_ids"]

            reps = model.get_patient_representation(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Ensure tensor is in a NumPy-compatible dtype (e.g. float32 instead of bfloat16).
            reps_np = reps.detach().to(torch.float32).cpu().numpy()

            if max_patients is not None:
                remaining = max_patients - processed
                if remaining <= 0:
                    break
                reps_np = reps_np[:remaining]
                patient_ids = patient_ids[:remaining]

            all_embeddings.append(reps_np)
            all_patient_ids.extend(patient_ids)
            processed += len(patient_ids)

            if max_patients is not None and processed >= max_patients:
                break

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted. Check dataset and filters.")

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_patient_ids


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model, tokenizer, data_config, _ = load_model_and_tokenizer(
        config_path=args.config_filepath,
        model_checkpoint=args.model_checkpoint,
        device=device,
    )

    print(f"Building dataloader for split='{args.split}'...")
    dataloader, dataset = build_dataloader(
        data_config=data_config,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
    )

    print("Extracting patient embeddings...")
    embeddings, patient_ids = extract_patient_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        max_patients=args.max_patients,
    )

    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(args.output_dir, "patient_ids.npy"), np.array(patient_ids, dtype=object))


if __name__ == "__main__":
    main()

