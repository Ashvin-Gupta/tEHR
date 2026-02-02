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
    Callback to verify packed sequences start and end with correct tokens.
    
    Checks a sample batch to ensure:
    - Sequences start with BOS token (or expected start token)
    - Sequences end with EOS token (between packed sequences)
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
        
        # Get a sample batch from the trainer
        trainer = kwargs.get('model')
        if trainer is None:
            print("  ⚠️  Could not access trainer for verification")
            return
        
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
            
            # Get token IDs
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            print(f"  - BOS token ID: {bos_token_id} ('{self.tokenizer.bos_token}')")
            print(f"  - EOS token ID: {eos_token_id} ('{self.tokenizer.eos_token}')")
            print(f"  - PAD token ID: {pad_token_id}\n")
            
            for i in range(min(self.num_samples, input_ids.size(0))):
                seq = input_ids[i]
                mask = attention_mask[i]
                
                # Find actual sequence (non-padding)
                non_pad_mask = mask.bool()
                if non_pad_mask.sum() == 0:
                    continue
                
                actual_seq = seq[non_pad_mask]
                seq_len = len(actual_seq)
                
                # Check start
                start_token_id = actual_seq[0].item()
                start_token_str = self.tokenizer.decode([start_token_id])
                
                # Check end
                end_token_id = actual_seq[-1].item()
                end_token_str = self.tokenizer.decode([end_token_id])
                
                # Show first and last few tokens
                first_5 = actual_seq[:5].tolist()
                last_5 = actual_seq[-5:].tolist()
                first_5_str = self.tokenizer.decode(first_5)
                last_5_str = self.tokenizer.decode(last_5)
                
                print(f"  Sequence {i+1}:")
                print(f"    Length: {seq_len} tokens")
                print(f"    Starts with: ID={start_token_id} ('{start_token_str}')")
                print(f"    Ends with:   ID={end_token_id} ('{end_token_str}')")
                print(f"    First 5 tokens: {first_5} → '{first_5_str[:100]}...'")
                print(f"    Last 5 tokens:  {last_5} → '...{last_5_str[-100:]}'")
                
                # Verify expectations
                if bos_token_id is not None and start_token_id != bos_token_id:
                    print(f"    ⚠️  WARNING: Expected BOS ({bos_token_id}), got {start_token_id}")
                if eos_token_id is not None and end_token_id != eos_token_id:
                    print(f"    ⚠️  WARNING: Expected EOS ({eos_token_id}), got {end_token_id}")
                print()
            
            # Check for EOS tokens within sequences (packing boundaries)
            print("  Checking for EOS tokens within sequences (packing boundaries)...")
            eos_positions = []
            for i in range(min(self.num_samples, input_ids.size(0))):
                seq = input_ids[i]
                mask = attention_mask[i]
                non_pad_seq = seq[mask.bool()]
                
                if eos_token_id is not None:
                    eos_pos = (non_pad_seq == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        eos_positions.append((i, eos_pos.tolist()))
            
            if eos_positions:
                print(f"  ✓ Found EOS tokens at positions: {eos_positions}")
                print("    (These mark boundaries between packed sequences)")
            else:
                print("  ⚠️  No EOS tokens found within sequences")
            
            self.verified = True
            
        except Exception as e:
            print(f"  ⚠️  Error during verification: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80 + "\n")
