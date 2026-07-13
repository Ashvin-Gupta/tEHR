"""
Training FLOPs estimator.

Uses the standard theoretical formula:

    FLOPs_train = 6 × N_params × T_tokens

where:
    6  = 2 (multiply-add counts as 2 ops) × 3 (backward ≈ 2× forward → total ≈ 3×)
    N_params = total model parameters (frozen + trainable)
    T_tokens = total_steps × effective_batch_size × seq_length
             = total_steps × (per_device_bs × grad_acc × world_size) × seq_length

References:
    - Kaplan et al. (2020) Scaling Laws for Neural Language Models
    - Hoffmann et al. (2022) Training Compute-Optimal Large Language Models
    - Brown et al. (2020) Language Models are Few-Shot Learners
"""
import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_parameters(model) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _human_readable(flops: float) -> str:
    """Format a raw FLOPs count as a human-readable string."""
    for unit, threshold in [("EFLOPs", 1e18), ("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9)]:
        if flops >= threshold:
            return f"{flops / threshold:.3f} {unit}"
    return f"{flops:.3e} FLOPs"


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def estimate_training_flops(
    model,
    total_steps: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    seq_length: int,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    Estimate total training FLOPs for a transformer model.

    Args:
        model: The model being trained (used for parameter counting).
        total_steps: Number of optimizer steps completed (trainer.state.global_step).
        per_device_batch_size: Batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation steps.
        seq_length: Sequence length (tokens per example — use max_length as upper bound).
        world_size: Number of GPUs / processes.

    Returns:
        Dict with FLOPs statistics.
    """
    params = _count_parameters(model)

    # Effective batch size = samples per optimizer step across all devices
    effective_batch = per_device_batch_size * gradient_accumulation_steps * world_size

    # Total tokens seen by the model during the full training run
    total_tokens = total_steps * effective_batch * seq_length

    # Forward-pass FLOPs per token ≈ 2 × N  (multiply-add = 2 ops)
    # Full training step (fwd + bwd) ≈ 6 × N per token  (backward ≈ 2× forward)
    training_flops = 6 * params["total"] * total_tokens

    return {
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "total_steps": total_steps,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "effective_batch_size": effective_batch,
        "seq_length": seq_length,
        "total_tokens_processed": total_tokens,
        "total_training_flops": training_flops,
        "total_training_flops_human": _human_readable(training_flops),
        "total_training_tflops": round(training_flops / 1e12, 4),
        "total_training_pflops": round(training_flops / 1e15, 8),
    }


def log_and_save_flops(
    flops_stats: Dict[str, Any],
    output_dir: Optional[str] = None,
    local_rank: int = 0,
):
    """
    Print the FLOPs summary to the log and optionally write a JSON file.

    Only rank-0 prints/saves to avoid duplicate output in multi-GPU runs.
    """
    if local_rank != 0:
        return

    sep = "=" * 80
    logger.info(f"\n{sep}")
    logger.info("TRAINING FLOPs SUMMARY")
    logger.info(sep)
    logger.info(f"  Total parameters:        {flops_stats['total_parameters']:,}")
    logger.info(f"  Trainable parameters:    {flops_stats['trainable_parameters']:,}")
    logger.info(f"  Total optimizer steps:   {flops_stats['total_steps']:,}")
    logger.info(f"  Per-device batch size:   {flops_stats['per_device_batch_size']}")
    logger.info(f"  Grad accumulation steps: {flops_stats['gradient_accumulation_steps']}")
    logger.info(f"  World size (GPUs):       {flops_stats['world_size']}")
    logger.info(f"  Effective batch size:    {flops_stats['effective_batch_size']}")
    logger.info(f"  Sequence length:         {flops_stats['seq_length']}")
    logger.info(f"  Total tokens processed:  {flops_stats['total_tokens_processed']:,}")
    logger.info(f"  Total training FLOPs:    {flops_stats['total_training_flops_human']}")
    logger.info(f"  Total training TFLOPs:   {flops_stats['total_training_tflops']:.4f}")
    logger.info(f"  Total training PFLOPs:   {flops_stats['total_training_pflops']:.8f}")
    logger.info(sep)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "training_flops.json")
        with open(save_path, "w") as fh:
            json.dump(flops_stats, fh, indent=2)
        logger.info(f"  FLOPs stats saved to: {save_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def report_flops_from_trainer(
    trainer,
    model,
    seq_length: int,
    output_dir: Optional[str] = None,
    local_rank: int = 0,
) -> Dict[str, Any]:
    """
    Compute and log FLOPs using information from a completed HF Trainer.

    Args:
        trainer:    A completed HuggingFace Trainer / SFTTrainer instance.
        model:      The model that was trained (for parameter counting).
        seq_length: Sequence length used in training (use max_length as upper bound).
        output_dir: Directory to write ``training_flops.json`` (None → skip file write).
        local_rank: Only rank 0 prints/saves.

    Returns:
        The FLOPs statistics dict.
    """
    total_steps = getattr(getattr(trainer, "state", None), "global_step", 0)
    args = getattr(trainer, "args", None)

    per_device_bs = getattr(args, "per_device_train_batch_size", 1) if args else 1
    grad_acc = getattr(args, "gradient_accumulation_steps", 1) if args else 1
    world_size = getattr(args, "world_size", 1) if args else 1

    flops_stats = estimate_training_flops(
        model=model,
        total_steps=total_steps,
        per_device_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc,
        seq_length=seq_length,
        world_size=world_size,
    )

    log_and_save_flops(flops_stats, output_dir=output_dir, local_rank=local_rank)
    return flops_stats
