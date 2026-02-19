"""
Baseline 4: Encoder Transformer

Uses transformer encoder on filtered EHR tokens. Uses the same filtering logic as XGBoost BOW:
- Includes: Medical events, lab test names (without values), measurement codes, demographics, lifestyle, special tokens
- Excludes: Time intervals, numeric values, units, lab value categories, quantile values, cancer-related tokens
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets, get_labels_from_dataset
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results
from src.baselines.baseline_xgboost_bow import should_include_token


def filter_tokens(token_list: List[str]) -> List[str]:
    """
    Filter out tokens using the same logic as XGBoost BOW baseline.
    Uses should_include_token function to filter tokens.
    """
    filtered = []
    for token in token_list:
        token_str = str(token)
        if should_include_token(token_str):
            filtered.append(token_str)
    return filtered


class FilteredTokenDataset(Dataset):
    """
    Dataset wrapper that filters tokens using the same logic as XGBoost BOW baseline.
    
    Uses should_include_token function to filter out:
    - Time intervals
    - Numeric values
    - Units
    - Lab value categories
    - Quantile values
    - Cancer-related tokens
    
    Keeps only:
    - Medical events (MEDICAL//...)
    - Lab test names (LAB//...) - without values
    - Measurement codes (MEASUREMENT//...)
    - Demographics (AGE, GENDER//..., ETHNICITY//..., REGION//...)
    - Lifestyle (LIFESTYLE//...)
    - Special tokens (<start>, <end>, <unknown>, MEDS_BIRTH)
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.filtered_samples = []
        
        # --- 1. PRE-CALCULATE BLACKLIST ---
        # Get the mapping from the underlying dataset
        id_to_token = getattr(base_dataset, 'id_to_token_map', {})
        exclude_ids = set()
        
        print("Identifying tokens to exclude using XGBoost BOW filtering logic...")
        print("  (Excluding: time, units, numbers, lab values, quantiles, cancer-related tokens)")
        for tid, token_str in id_to_token.items():
            token_str = str(token_str)
            
            # Use the same filtering logic as XGBoost BOW
            # Exclude tokens that don't pass should_include_token check
            # Also exclude cancer-related tokens explicitly
            if not should_include_token(token_str) or 'cancer' in token_str.lower():
                exclude_ids.add(tid)
        
        # --- 2. FILTER SEQUENCES USING BLACKLIST ---
        print(f"Filtering {len(base_dataset)} samples...")
        samples_without_tokens = 0
        samples_filtered_empty = 0
        
        for i in tqdm(range(len(base_dataset)), desc="Filtering"):
            sample = base_dataset[i]
            
            # If sample is None, UnifiedEHRDataset couldn't find a label for this patient
            if sample is None:
                continue
                
            if 'tokens' not in sample:
                samples_without_tokens += 1
                if samples_without_tokens <= 5:  # Print first few for debugging
                    print(f"  Warning: Sample {i} doesn't have 'tokens' key. Keys: {sample.keys()}")
                continue
            
            token_ids = sample['tokens']
            
            # Handle both list and tensor formats
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            # Fast filtering using the ID blacklist
            # This keeps only IDs NOT in our exclude set
            filtered_ids = [tid for tid in token_ids if tid not in exclude_ids]
            
            # Re-assign the filtered IDs
            sample['tokens'] = filtered_ids
            
            # Only keep the sample if it still has events left
            if len(filtered_ids) > 0:
                self.filtered_samples.append(sample)
            else:
                samples_filtered_empty += 1
        
        print(f"  - Excluded {len(exclude_ids):,} token IDs from vocabulary")
        print(f"  - Kept {len(self.filtered_samples):,} samples after filtering")
        if samples_without_tokens > 0:
            print(f"  - Warning: {samples_without_tokens} samples didn't have 'tokens' key")
        if samples_filtered_empty > 0:
            print(f"  - Warning: {samples_filtered_empty} samples were filtered to empty sequences")
        
        if len(self.filtered_samples) == 0:
            print("ERROR: All samples were filtered out or skipped!")
            print("  This could mean:")
            print("    1. All tokens were excluded by the filter")
            print("    2. Samples don't have 'tokens' key (check dataset format)")
            print("    3. Dataset is empty or has no valid labels")
    
    def __len__(self):
        return len(self.filtered_samples)
    
    def __getitem__(self, idx):
        return self.filtered_samples[idx]


class EHRTransformerEncoder(nn.Module):
    """
    Transformer encoder for EHR token sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_labels: int = 2
    ):
        """
        Initialize transformer encoder.
        
        Args:
            vocab_size: Size of token vocabulary
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            num_labels: Number of output classes
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head (logistic regression-style)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode token IDs into contextual representations (no pooling or head).
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Sequence output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Truncate if sequence exceeds max_seq_length (safety check)
        if seq_len > self.max_seq_length:
            token_ids = token_ids[:, :self.max_seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_length]
            seq_len = self.max_seq_length
        
        # Embeddings
        x = self.token_embedding(token_ids)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding (truncate pos_encoding if needed)
        pos_len = min(seq_len, self.max_seq_length)
        x = x + self.pos_encoding[:pos_len, :].unsqueeze(0)
        x = self.dropout(x)
        
        # Create attention mask (invert: 1 = attend, 0 = ignore)
        if attention_mask is not None:
            # Convert to transformer format: True = ignore, False = attend
            mask = (attention_mask == 0)
        else:
            mask = None
        
        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)  # [batch, seq_len, hidden_dim]
        return x
    
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, num_labels]
        """
        x = self.encode(token_ids, attention_mask)
        
        # Pool: use mean pooling over sequence
        if attention_mask is not None:
            mask_expanded = attention_mask.float().unsqueeze(-1)
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Classify
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        return logits


class EHRTransformerMLMHead(nn.Module):
    """
    MLM head that wraps an encoder and predicts token IDs.
    """
    
    def __init__(self, encoder: EHRTransformerEncoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(encoder.hidden_dim, vocab_size)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] with -100 at non-MLM positions
        
        Returns:
            loss (optional), logits [batch, seq_len, vocab_size]
        """
        sequence_output = self.encoder.encode(token_ids, attention_mask)
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )
        return loss, logits


class EncoderTransformerBaseline:
    """
    Encoder transformer baseline for EHR classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder transformer baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/encoder_transformer')
        
        self.model = None
        self.mlm_model = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        # max_seq_length will be set when model is created
        self.max_seq_length = None
        self._label_conversion_warned = False  # Track if we've warned about label conversion
        self.pretrain_config = config.get('pretrain', {})  # Optional stage-1 config
    
    def build_vocabulary(self, datasets: Dict[str, Dataset]):
        """
        Build vocabulary from all datasets.
        
        Args:
            datasets: Dictionary of datasets
        """
        print("\n" + "=" * 80)
        print("Building vocabulary from datasets...")
        print("=" * 80)
        
        # Count token frequencies across all datasets
        token_counter: Counter = Counter()
        
        for split_name, dataset in datasets.items():
            print(f"  - Processing {split_name} split...")
            for i in tqdm(range(len(dataset)), desc=f"Collecting tokens from {split_name}"):
                sample = dataset[i]
                if sample is not None and 'tokens' in sample:
                    tokens = sample['tokens']
                    for token in tokens:
                        if isinstance(token, (int, str)):
                            token_str = str(token)
                            token_counter[token_str] += 1
        
        # Report total unique tokens before truncation
        num_unique_tokens = len(token_counter)
        print(f"  - Total unique tokens observed: {num_unique_tokens:,}")
        
        # Keep only the top K most frequent tokens
        max_vocab_tokens = 5000
        most_common_tokens = token_counter.most_common(max_vocab_tokens)
        kept_tokens = [tok for tok, _ in most_common_tokens]
        
        print(f"  - Keeping top {len(kept_tokens):,} most frequent tokens for vocabulary")
        
        # Create vocabulary from kept tokens only
        kept_tokens = sorted(kept_tokens)
        # Reserve special tokens: PAD, UNK, MASK
        vocab_size = len(kept_tokens) + 3
        
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<MASK>': 2}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<MASK>'}
        
        for i, token in enumerate(kept_tokens, start=3):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.vocab = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'size': vocab_size
        }
        
        print(f"  - Final vocabulary size (including PAD/UNK): {vocab_size:,}")
    
    def tokenize_sequence(self, tokens: List) -> torch.Tensor:
        """
        Convert token sequence to IDs.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Tensor of token IDs
        """
        token_ids = []
        for token in tokens:
            token_str = str(token)
            # Only keep tokens that are in the top-K vocabulary.
            # Tokens not in vocab are removed from the stream.
            if token_str in self.token_to_id:
                token_ids.append(self.token_to_id[token_str])
        
        # If all tokens were filtered out for this sequence, return a single PAD token
        # so that downstream code always sees at least one token.
        if len(token_ids) == 0:
            token_ids = [self.token_to_id['<PAD>']]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def mlm_collate_fn(self, batch):
        """
        Collate function for MLM pretraining.
        
        Returns input_ids, attention_mask, and MLM labels.
        """
        base_batch = self.collate_fn(batch)
        input_ids = base_batch['token_ids']
        attention_mask = base_batch['attention_mask']
        
        labels = input_ids.clone()
        
        # Do not predict on padding tokens
        pad_id = self.token_to_id['<PAD>']
        special_mask = input_ids.eq(pad_id)
        
        mlm_probability = float(self.pretrain_config.get('mlm_probability', 0.15))
        probability_matrix = torch.full(labels.shape, mlm_probability)
        probability_matrix.masked_fill_(special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        
        mask_token_id = self.token_to_id.get('<MASK>', self.token_to_id['<UNK>'])
        
        # 80% of selected tokens → [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id
        
        # 10% → random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.token_to_id), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% → keep original token (already handled by masked_indices but not in indices_replaced/indices_random)
        
        return {
            'token_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples
        
        Returns:
            Batched tensors
        """
        # Get max_seq_length (default to 512 if not set)
        max_seq_length = self.max_seq_length if self.max_seq_length is not None else 512
        
        token_ids_list = []
        labels_list = []
        max_len = 0
        
        for sample in batch:
            if sample is not None:
                tokens = sample.get('tokens', [])
                token_ids = self.tokenize_sequence(tokens)
                
                # Truncate to max_seq_length if needed
                if len(token_ids) > max_seq_length:
                    token_ids = token_ids[:max_seq_length]
                
                token_ids_list.append(token_ids)
                max_len = max(max_len, len(token_ids))
                
                label = sample['label']
                if hasattr(label, 'item'):
                    label = label.item()
                
                # Convert multi-class labels to binary: 0 = Control, >0 = Cancer (1)
                # The dataset returns multi-class labels (0-19), but model expects binary (0 or 1)
                binary_label = 0 if label == 0 else 1
                
                # Debug: print first conversion example
                if not self._label_conversion_warned and len(labels_list) == 0:
                    print(f"  Converting labels: multi-class label {label} -> binary label {binary_label}")
                    self._label_conversion_warned = True
                
                labels_list.append(binary_label)
        
        # Ensure max_len doesn't exceed max_seq_length
        max_len = min(max_len, max_seq_length)
        
        # Pad sequences
        batch_size = len(token_ids_list)
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, token_ids in enumerate(token_ids_list):
            seq_len = min(len(token_ids), max_len)
            padded_ids[i, :seq_len] = token_ids[:seq_len]
            attention_mask[i, :seq_len] = 1
        
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Verify binary labels
        unique_labels = torch.unique(labels)
        if len(unique_labels) > 2 or not all(l.item() in [0, 1] for l in unique_labels):
            print(f"Warning: Expected binary labels [0, 1], got {unique_labels.tolist()}")
        
        return {
            'token_ids': padded_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def create_model(self):
        """Create transformer encoder model."""
        print("\n" + "=" * 80)
        print("Creating transformer encoder model...")
        print("=" * 80)
        
        vocab_size = int(self.vocab['size'])
        hidden_dim = int(self.model_config.get('hidden_dim', 512))
        num_layers = int(self.model_config.get('num_layers', 6))
        num_heads = int(self.model_config.get('num_heads', 8))
        ff_dim = int(self.model_config.get('ff_dim', 2048))
        dropout = float(self.model_config.get('dropout', 0.1))
        max_seq_length = int(self.model_config.get('max_seq_length', 512))
        num_labels = int(self.model_config.get('num_labels', 2))
        
        # Store max_seq_length for use in collate_fn
        self.max_seq_length = max_seq_length
        
        self.model = EHRTransformerEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_labels=num_labels
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def create_mlm_model(self):
        """Create MLM head that wraps the encoder."""
        if self.model is None:
            self.create_model()
        
        vocab_size = int(self.vocab['size'])
        self.mlm_model = EHRTransformerMLMHead(self.model, vocab_size=vocab_size)
    
    def train_mlm_epoch(self, train_loader, optimizer, device):
        """Train MLM for one epoch."""
        self.mlm_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="MLM Training"):
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss, _ = self.mlm_model(token_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate_mlm(self, data_loader, device):
        """Compute average MLM loss on validation set."""
        self.mlm_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="MLM Eval"):
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, _ = self.mlm_model(token_ids, attention_mask, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train_epoch(self, train_loader, optimizer, criterion, device):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = self.model(token_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader, device, split_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader instance
            device: Device to run on
            split_name: Name of the split
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.model(token_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_labels.extend(labels_np)
                all_probs.extend(probs)
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = compute_baseline_metrics(all_labels, all_probs)
        
        # Print results
        print_results(metrics, split_name)
        
        # Save all plots (ROC, PR, and Calibration)
        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(all_labels, all_probs, plot_dir)
        
        # Save predictions
        results = {
            'metrics': metrics,
            'labels': all_labels.tolist(),
            'probs': all_probs.tolist()
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        return metrics
    
    def run(self):
        """Run the complete baseline training and evaluation pipeline."""
        # Setup
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='tokens'
        )
        
        # Filter tokens
        filtered_datasets = {}
        for split_name, dataset in datasets.items():
            filtered_datasets[split_name] = FilteredTokenDataset(dataset)
        
        # Build vocabulary
        self.build_vocabulary(filtered_datasets)
        
        # Create model
        self.create_model()
        self.model = self.model.to(device)
        
        # Optional Stage 1: MLM pretraining on encoder
        if self.pretrain_config.get('enabled', False):
            print("\n" + "=" * 80)
            print("Stage 1: MLM pretraining for encoder transformer")
            print("=" * 80)
            
            # Freeze classification head during MLM
            for p in self.model.classifier.parameters():
                p.requires_grad = False
            
            self.create_mlm_model()
            self.mlm_model = self.mlm_model.to(device)
            
            mlm_batch_size = int(self.pretrain_config.get('batch_size', self.training_config.get('batch_size', 32)))
            mlm_train_loader = DataLoader(
                filtered_datasets['train'],
                batch_size=mlm_batch_size,
                shuffle=True,
                collate_fn=self.mlm_collate_fn,
                num_workers=int(self.pretrain_config.get('dataloader_num_workers', self.training_config.get('dataloader_num_workers', 4)))
            )
            mlm_val_loader = DataLoader(
                filtered_datasets['tuning'],
                batch_size=mlm_batch_size,
                shuffle=False,
                collate_fn=self.mlm_collate_fn,
                num_workers=int(self.pretrain_config.get('dataloader_num_workers', self.training_config.get('dataloader_num_workers', 4)))
            )
            
            mlm_optimizer = torch.optim.AdamW(
                self.mlm_model.parameters(),
                lr=float(self.pretrain_config.get('learning_rate', self.training_config.get('learning_rate', 1e-4))),
                weight_decay=float(self.pretrain_config.get('weight_decay', self.training_config.get('weight_decay', 0.01)))
            )
            mlm_epochs = int(self.pretrain_config.get('epochs', 1))
            
            best_mlm_loss = float('inf')
            for epoch in range(mlm_epochs):
                print(f"\n[MLM] Epoch {epoch + 1}/{mlm_epochs}")
                train_mlm_loss = self.train_mlm_epoch(mlm_train_loader, mlm_optimizer, device)
                val_mlm_loss = self.evaluate_mlm(mlm_val_loader, device)
                print(f"  - MLM train loss: {train_mlm_loss:.4f}")
                print(f"  - MLM val loss:   {val_mlm_loss:.4f}")
                best_mlm_loss = min(best_mlm_loss, val_mlm_loss)
            
            print(f"\nFinished MLM pretraining. Best val loss: {best_mlm_loss:.4f}")
            
            # Re-enable classification head for supervised fine-tuning
            for p in self.model.classifier.parameters():
                p.requires_grad = True
        
        # Create data loaders for supervised training
        batch_size = int(self.training_config.get('batch_size', 32))
        train_loader = DataLoader(
            filtered_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=int(self.training_config.get('dataloader_num_workers', 4))
        )
        val_loader = DataLoader(
            filtered_datasets['tuning'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=int(self.training_config.get('dataloader_num_workers', 4))
        )
        test_loader = DataLoader(
            filtered_datasets['held_out'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=int(self.training_config.get('dataloader_num_workers', 4))
        )
        
        # Training setup for supervised classification
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_config.get('learning_rate', 1e-4)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01))
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train
        num_epochs = int(self.training_config.get('epochs', 10))
        print(f"\n" + "=" * 80)
        print(f"Stage 2: Supervised training for {num_epochs} epochs...")
        print("=" * 80)
        
        best_val_auroc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n[Classification] Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader, optimizer, criterion, device)
            print(f"  - Train loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.evaluate(val_loader, device, 'validation')
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'models', 'best_model.pt'))
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'models', 'best_model.pt')))
        
        # Final evaluation
        val_metrics = self.evaluate(val_loader, device, 'validation')
        test_metrics = self.evaluate(test_loader, device, 'test')
        
        # Save summary
        summary = {
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')
        
        print("\n" + "=" * 80)
        print("Encoder Transformer Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

