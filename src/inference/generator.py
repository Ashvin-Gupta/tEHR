"""
Inference utilities for EHR language models.
"""
import textwrap
from threading import Thread
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel


def run_ehr_inference(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 256, 
    max_print_width: int = 100,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 30,
    repetition_penalty: float = 1.2
):
    """
    Runs causal language modeling inference on the fine-tuned model and displays
    the raw stream and formatted event output.
    
    Args:
        model: The fine-tuned LoRA model (e.g., Unsloth FastLanguageModel).
        tokenizer: The corresponding tokenizer.
        prompt: The input medical event sequence.
        max_new_tokens: Maximum number of tokens to generate.
        max_print_width: Width for text wrapping the final output.
        temperature: Sampling temperature (lower = more conservative).
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling parameter.
        repetition_penalty: Penalty for repeating tokens.
    
    Returns:
        str: The generated text.
    """
    # Setup for inference
    model.eval()
    FastLanguageModel.for_inference(model)

    # Encode the prompt and move to the correct device
    device = model.device 
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print("-" * 80)
    print(f"PROMPT:\n{textwrap.fill(prompt, width=max_print_width)}\n")
    print("MODEL GENERATION:")
    print("-" * 80)

    # Setup the streamer
    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    all_generated_chunks = []
    
    # Setup generation arguments
    generation_kwargs = dict(
        inputs,
        streamer=text_streamer,
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
    
    # Start generation thread and stream raw output
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("MODEL OUTPUT (RAW STREAM):")
    for new_text in text_streamer:
        print(new_text, end="")
        all_generated_chunks.append(new_text)

    thread.join()
    print("\n")
    
    return "".join(all_generated_chunks)
