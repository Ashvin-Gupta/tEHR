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
