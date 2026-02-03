"""
Baseline 1: Pure LLM Zero-Shot Approach

Uses base Qwen model without any training, just prompting for cancer risk prediction.
"""
import os
import re
import torch
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
from unsloth import FastLanguageModel

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets, get_labels_from_dataset
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results


class PureLLMBaseline:
    """
    Zero-shot LLM baseline for cancer prediction.
    
    Uses base model with simple prompt to extract risk probability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pure LLM baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.output_dir = config.get('output_dir', './outputs/pure_llm')
        
        self.model = None
        self.tokenizer = None
        self.prompt_template = "Given this patient's medical history: {ehr_text}\n\nWhat is the risk of pancreatic cancer? Answer with a probability between 0 and 1."
    
    def load_model(self):
        """Load base Qwen model (no pretraining, no LoRA)."""
        print("\n" + "=" * 80)
        print("Loading base Qwen model...")
        print("=" * 80)
        
        model_name = self.model_config.get('model_name', 'unsloth/Qwen2.5-7B-Instruct')
        max_length = self.model_config.get('max_length', 8192)
        load_in_4bit = self.model_config.get('load_in_4bit', True)
        
        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - 4-bit quantization: {load_in_4bit}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto"
        )
        
        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")
    
    def format_prompt(self, ehr_text: str) -> str:
        """
        Format EHR text with prompt template.
        
        Args:
            ehr_text: Patient EHR text
        
        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(ehr_text=ehr_text)
    
    def extract_probability(self, output_text: str) -> float:
        """
        Extract probability from model output.
        
        Tries multiple strategies to parse probability from text.
        
        Args:
            output_text: Model generated text
        
        Returns:
            Extracted probability (0.0 to 1.0), or 0.5 if parsing fails
        """
        # Strategy 1: Look for explicit probability value
        # Pattern: "0.XX" or "0.XXX" or "0.X"
        prob_patterns = [
            r'\b0\.\d{1,3}\b',  # 0.5, 0.75, 0.123
            r'\b\d+%',  # 50%, 75%
            r'probability[:\s]+([0-9.]+)',
            r'risk[:\s]+([0-9.]+)',
        ]
        
        for pattern in prob_patterns:
            matches = re.findall(pattern, output_text.lower())
            if matches:
                try:
                    value = float(matches[0].replace('%', ''))
                    if '%' in matches[0]:
                        value = value / 100.0
                    if 0.0 <= value <= 1.0:
                        return value
                except:
                    continue
        
        # Strategy 2: Look for percentage
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', output_text)
        if percent_match:
            try:
                value = float(percent_match.group(1)) / 100.0
                if 0.0 <= value <= 1.0:
                    return value
            except:
                pass
        
        # Strategy 3: Look for fraction
        fraction_match = re.search(r'(\d+)/(\d+)', output_text)
        if fraction_match:
            try:
                num = float(fraction_match.group(1))
                den = float(fraction_match.group(2))
                if den > 0:
                    value = num / den
                    if 0.0 <= value <= 1.0:
                        return value
            except:
                pass
        
        # Default: return 0.5 (neutral)
        print(f"Warning: Could not extract probability from: {output_text[:100]}...")
        return 0.5
    
    def predict_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generate predictions for a batch of texts.
        
        Args:
            texts: List of EHR texts
            batch_size: Batch size for inference
        
        Returns:
            Array of predicted probabilities
        """
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating predictions"):
                batch_texts = texts[i:i+batch_size]
                batch_prompts = [self.format_prompt(text) for text in batch_texts]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.get('max_length', 8192)
                ).to(self.model.device)
                
                # Generate
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode and extract probabilities
                for j, output_ids in enumerate(outputs):
                    # Remove input tokens
                    generated_ids = output_ids[len(inputs['input_ids'][j]):]
                    output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    prob = self.extract_probability(output_text)
                    probs.append(prob)
        
        return np.array(probs)
    
    def evaluate(self, dataset, split_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: UnifiedEHRDataset instance
            split_name: Name of the split
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n" + "=" * 80)
        print(f"Evaluating on {split_name} set...")
        print("=" * 80)
        
        # Extract texts and labels
        texts = []
        labels = []
        
        print("Loading data...")
        for i in tqdm(range(len(dataset)), desc="Loading samples"):
            sample = dataset[i]
            if sample is not None:
                texts.append(sample['text'])
                label = sample['label'].item() if hasattr(sample['label'], 'item') else sample['label']
                labels.append(label)
        
        labels = np.array(labels)
        print(f"  - Loaded {len(texts)} samples")
        
        # Generate predictions
        batch_size = self.config.get('inference', {}).get('batch_size', 4)
        probs = self.predict_batch(texts, batch_size=batch_size)
        
        # Compute metrics
        metrics = compute_baseline_metrics(labels, probs)
        
        # Print results
        print_results(metrics, split_name)
        
        # Save all plots (ROC, PR, and Calibration)
        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(labels, probs, plot_dir)
        
        # Save predictions
        results = {
            'metrics': metrics,
            'labels': labels.tolist(),
            'probs': probs.tolist()
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        return metrics
    
    def run(self):
        """Run the complete baseline evaluation pipeline."""
        # Setup
        setup_output_dir(self.output_dir, overwrite=self.config.get('overwrite_output_dir', False))
        
        # Load model
        self.load_model()
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['tuning', 'held_out'],
            format='text'
        )
        
        # Evaluate on validation and test sets
        val_metrics = self.evaluate(datasets['tuning'], 'validation')
        test_metrics = self.evaluate(datasets['held_out'], 'test')
        
        # Save summary
        summary = {
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')
        
        print("\n" + "=" * 80)
        print("Pure LLM Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

