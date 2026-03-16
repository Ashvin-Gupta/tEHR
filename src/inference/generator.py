"""
Inference utilities for EHR language models.

This version performs a single non‑streaming generation call and returns only
the decoded continuation text. Printing is delegated to the caller so that
logs do not contain duplicate outputs.
"""
from unsloth import FastLanguageModel


def run_ehr_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 30,
    repetition_penalty: float = 1.2,
):
    """
    Runs causal language modeling inference on the fine‑tuned model and returns
    the generated continuation text.

    Args:
        model: The fine‑tuned model (e.g., Unsloth FastLanguageModel).
        tokenizer: The corresponding tokenizer.
        prompt: The input medical event sequence (full prompt string).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more conservative).
        top_p: Nucleus sampling probability threshold.
        top_k: Top‑k sampling parameter.
        repetition_penalty: Penalty for repeating tokens.

    Returns:
        str: The generated continuation (decoded text only, without the prompt).
    """
    model.eval()
    FastLanguageModel.for_inference(model)

    device = model.device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs = model.generate(**generation_kwargs)

    # Remove the prompt tokens so we only return the continuation.
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
