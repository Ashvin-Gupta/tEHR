"""
Training callbacks for EHR model training.
"""
from transformers import TrainerCallback
from src.inference.generator import run_ehr_inference


class InferenceCallback(TrainerCallback):
    """
    Callback to run inference tests at the end of each epoch.
    
    This allows monitoring of model generation quality during training.
    """
    
    def __init__(self, model, tokenizer, prompt: str, **inference_kwargs):
        """
        Args:
            model: The model being trained.
            tokenizer: The tokenizer for the model.
            prompt: The prompt to use for inference tests.
            **inference_kwargs: Additional arguments for run_ehr_inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.inference_kwargs = inference_kwargs
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Run inference test at the end of each epoch."""
        print("\n" + "=" * 80)
        print(f"Running inference test after epoch {state.epoch}...")
        print("=" * 80)
        run_ehr_inference(
            self.model, 
            self.tokenizer, 
            self.prompt,
            **self.inference_kwargs
        )
        print("\n")

class PackingVerificationCallback(TrainerCallback):
    """
    Callback to verify packed sequences have correct EOS token placement.
    
    Verifies that:
    - Number of EOS tokens = Number of patients - 1
    (EOS appears between patients, not after the last one)
    """
    
    def __init__(self, tokenizer, num_samples=3):
        """
        Args:
            tokenizer: The tokenizer to decode tokens.
            num_samples: Number of sample sequences to check.
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.verified = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Verify packed sequences at the start of training."""
        if self.verified:
            return
        
        print("\n" + "=" * 80)
        print("VERIFYING PACKED SEQUENCES")
        print("=" * 80)
        
        # Get token IDs
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Get <start> token ID(s) - encode the string to find its tokenization
        start_token_str = "<start>"
        start_token_ids = self.tokenizer.encode(start_token_str, add_special_tokens=False)
        # Usually <start> will tokenize to a single token, but handle multiple tokens
        if len(start_token_ids) == 0:
            print(f"  ⚠️  WARNING: Could not find token ID for '<start>' token")
            return
        
        print(f"  - BOS token ID: {bos_token_id} ('{self.tokenizer.bos_token}')")
        print(f"  - EOS token ID: {eos_token_id} ('{self.tokenizer.eos_token}')")
        print(f"  - <start> token ID(s): {start_token_ids}")
        print(f"  - PAD token ID: {pad_token_id}\n")
        
        # Access the trainer's dataloader
        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is None:
            print("  ⚠️  Could not access dataloader for verification")
            return
        
        # Get first batch
        try:
            batch = next(iter(train_dataloader))
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            print(f"  - Batch shape: {input_ids.shape}")
            print(f"  - Checking {min(self.num_samples, input_ids.size(0))} sequences\n")
            
            for i in range(min(self.num_samples, input_ids.size(0))):
                seq = input_ids[i]
                mask = attention_mask[i]
                
                # Find actual sequence (non-padding)
                non_pad_mask = mask.bool()
                if non_pad_mask.sum() == 0:
                    continue
                
                actual_seq = seq[non_pad_mask]
                seq_len = len(actual_seq)
                
                # Count patients: look for <start> token pattern
                # <start> might be tokenized as multiple tokens, so we need to check for the pattern
                # For simplicity, check if the first token ID appears (most common case)
                patient_count = 0
                start_pattern = start_token_ids[0]  # Use first token ID as marker
                
                # Count occurrences of start token ID
                patient_count = (actual_seq == start_pattern).sum().item()
                
                # Count EOS tokens
                eos_count = 0
                if eos_token_id is not None:
                    eos_count = (actual_seq == eos_token_id).sum().item()
                    eos_positions = (actual_seq == eos_token_id).nonzero(as_tuple=True)[0].tolist()
                else:
                    eos_positions = []
                
                # Verify: EOS_count should equal patient_count - 1
                expected_eos_count = max(0, patient_count - 1)
                
                print(f"  Sequence {i+1}:")
                print(f"    Length: {seq_len} tokens")
                print(f"    Number of patients (counted by <start> tokens): {patient_count}")
                print(f"    Number of EOS tokens: {eos_count}")
                print(f"    Expected EOS tokens: {expected_eos_count} (patients - 1)")
                
                if eos_count == expected_eos_count:
                    print(f"    ✓ CORRECT: EOS count matches expected ({eos_count} = {patient_count} - 1)")
                else:
                    print(f"    ⚠️  WARNING: EOS count mismatch! Got {eos_count}, expected {expected_eos_count}")
                    if eos_count < expected_eos_count:
                        print(f"      → Missing {expected_eos_count - eos_count} EOS token(s)")
                    else:
                        print(f"      → Found {eos_count - expected_eos_count} extra EOS token(s)")
                
                if eos_positions:
                    print(f"    EOS token positions: {eos_positions[:10]}{'...' if len(eos_positions) > 10 else ''}")
                
                # Show sample of sequence structure
                first_10 = actual_seq[:10].tolist()
                first_10_str = self.tokenizer.decode(first_10)
                print(f"    First 10 tokens: {first_10} → '{first_10_str[:80]}...'")
                print()
            
            self.verified = True
            
        except Exception as e:
            print(f"  ⚠️  Error during verification: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80 + "\n")

class BatchShapeCallback(TrainerCallback):
    """
    Callback to print batch shapes during training for debugging.
    """
    
    def __init__(self, print_every_n_steps=10):
        """
        Args:
            print_every_n_steps: Print batch info every N steps (0 = only first batch).
        """
        self.print_every_n_steps = print_every_n_steps
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Print batch shape at the start of each step."""
        if self.print_every_n_steps == 0 and self.step_count > 0:
            return  # Only print first batch
        
        if self.print_every_n_steps > 0 and self.step_count % self.print_every_n_steps != 0:
            self.step_count += 1
            return
        
        # Try to get batch info from kwargs
        inputs = kwargs.get('inputs', {})
        if inputs:
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                print(f"\n[Step {state.global_step}] Batch shape: {input_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")
                
                if attention_mask is not None:
                    # Count actual non-padding tokens
                    actual_lengths = attention_mask.sum(dim=1)
                    print(f"  - Actual sequence lengths: min={actual_lengths.min().item()}, max={actual_lengths.max().item()}, mean={actual_lengths.float().mean().item():.1f}")
        
        self.step_count += 1
