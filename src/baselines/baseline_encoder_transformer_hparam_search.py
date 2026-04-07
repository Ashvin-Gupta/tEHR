import copy
import itertools
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baselines.baseline_encoder_transformer import (
    EHRTransformerEncoder,
    EHRTransformerMLMHead,
    FilteredTokenDataset,
)
from src.baselines.baseline_xgboost_bow import should_include_token
from src.baselines.baseline_utils import setup_output_dir
from src.data.dataset_factory import get_dataset
from src.evaluations.evaluations import (
    compute_baseline_metrics,
    plot_all_curves,
    print_results,
    save_results,
)


class EncoderTransformerHyperparamSearchBaseline:
    """
    Hyperparameter search for Transformer encoder baseline.
    Selects best hyperparameters on tuning and evaluates once on held_out.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config["data"]
        self.training_config = config.get("training", {})
        self.pretrain_config = config.get("pretrain", {})
        self.search_config = config.get("hyperparameter_search", {})

        self.output_dir = self.training_config.get(
            "output_dir", "./outputs/encoder_transformer_hparam_search"
        )

        self.model = None
        self.mlm_model = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        self.max_seq_length = None

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_datasets(self) -> Dict[str, Any]:
        dataset_name = self.data_config.get("dataset", "CPRD")
        data_cfg = dict(self.data_config)
        data_cfg["format"] = "tokens"
        return {
            split: get_dataset(dataset_name, data_cfg, split)
            for split in ["train", "tuning", "held_out"]
        }

    def build_vocabulary(self, datasets: Dict[str, Any]) -> None:
        token_counter = {}
        max_vocab_tokens = int(self.model_config.get("max_vocab_tokens", 5000))

        for split_name, dataset in datasets.items():
            print(f"Collecting tokens from {split_name}...")
            for i in tqdm(range(len(dataset)), desc=f"Vocab {split_name}"):
                sample = dataset[i]
                if sample is None or "tokens" not in sample:
                    continue
                for token in sample["tokens"]:
                    token_str = str(token)
                    if should_include_token(token_str) and "cancer" not in token_str.lower():
                        token_counter[token_str] = token_counter.get(token_str, 0) + 1

        kept = sorted(
            [tok for tok, _ in sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:max_vocab_tokens]]
        )
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<MASK>"}
        for i, token in enumerate(kept, start=3):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        self.vocab = {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token, "size": len(kept) + 3}
        print(f"Vocabulary size: {self.vocab['size']:,}")

    def tokenize_sequence(self, tokens: List[Any]) -> torch.Tensor:
        token_ids = []
        for token in tokens:
            token_str = str(token)
            if token_str in self.token_to_id:
                token_ids.append(self.token_to_id[token_str])
        if not token_ids:
            token_ids = [self.token_to_id["<PAD>"]]
        return torch.tensor(token_ids, dtype=torch.long)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_seq_length = self.max_seq_length if self.max_seq_length is not None else 512
        token_ids_list = []
        labels_list = []
        max_len = 0

        for sample in batch:
            if sample is None:
                continue
            token_ids = self.tokenize_sequence(sample.get("tokens", []))
            if len(token_ids) > max_seq_length:
                token_ids = token_ids[:max_seq_length]
            token_ids_list.append(token_ids)
            max_len = max(max_len, len(token_ids))

            label = sample["label"]
            label = label.item() if hasattr(label, "item") else int(label)
            labels_list.append(0 if label == 0 else 1)

        max_len = min(max_len, max_seq_length)
        batch_size = len(token_ids_list)
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, token_ids in enumerate(token_ids_list):
            seq_len = min(len(token_ids), max_len)
            padded_ids[i, :seq_len] = token_ids[:seq_len]
            attention_mask[i, :seq_len] = 1

        return {
            "token_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    def mlm_collate_fn(self, batch: List[Dict[str, Any]], mlm_probability: float) -> Dict[str, torch.Tensor]:
        base_batch = self.collate_fn(batch)
        input_ids = base_batch["token_ids"]
        attention_mask = base_batch["attention_mask"]
        labels = input_ids.clone()

        pad_id = self.token_to_id["<PAD>"]
        special_mask = input_ids.eq(pad_id)
        probability_matrix = torch.full(labels.shape, float(mlm_probability))
        probability_matrix.masked_fill_(special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        mask_token_id = self.token_to_id["<MASK>"]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.token_to_id), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"token_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _build_model(self, trial_cfg: Dict[str, Any]) -> EHRTransformerEncoder:
        hidden_dim = int(trial_cfg["hidden_dim"])
        num_layers = int(trial_cfg["num_layers"])
        num_heads = int(trial_cfg["num_heads"])
        self.max_seq_length = int(self.model_config.get("max_seq_length", 512))
        ff_dim = int(self.model_config.get("ff_dim", hidden_dim * 4))
        dropout = float(self.model_config.get("dropout", 0.1))
        num_labels = int(self.model_config.get("num_labels", 2))

        return EHRTransformerEncoder(
            vocab_size=int(self.vocab["size"]),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_length=self.max_seq_length,
            num_labels=num_labels,
        )

    def _eval_classifier(self, data_loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in data_loader:
                token_ids = batch["token_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = self.model(token_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        labels_np = np.array(all_labels)
        probs_np = np.array(all_probs)
        return compute_baseline_metrics(labels_np, probs_np), labels_np, probs_np

    def _eval_mlm_loss(self, data_loader: DataLoader, device: torch.device) -> float:
        self.mlm_model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in data_loader:
                token_ids = batch["token_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                loss, _ = self.mlm_model(token_ids, attention_mask, labels)
                total_loss += float(loss.item())
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _run_trial(
        self,
        trial_cfg: Dict[str, Any],
        train_dataset: Any,
        tuning_dataset: Any,
        held_out_dataset: Any,
        device: torch.device,
    ) -> Dict[str, Any]:
        self.model = self._build_model(trial_cfg).to(device)
        batch_size = int(self.training_config.get("batch_size", 32))
        num_workers = int(self.training_config.get("dataloader_num_workers", 2))

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=num_workers
        )
        tuning_loader = DataLoader(
            tuning_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=num_workers
        )
        held_out_loader = DataLoader(
            held_out_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=num_workers
        )

        if self.pretrain_config.get("enabled", True):
            mlm_prob = float(trial_cfg["mlm_probability"])
            mlm_batch_size = int(self.pretrain_config.get("batch_size", batch_size))
            pretrain_epochs = int(self.pretrain_config.get("epochs", 20))
            pretrain_patience = int(self.pretrain_config.get("early_stopping_patience", 3))
            pretrain_min_delta = float(self.pretrain_config.get("early_stopping_min_delta", 1e-4))

            self.mlm_model = EHRTransformerMLMHead(self.model, vocab_size=int(self.vocab["size"])).to(device)
            mlm_optimizer = torch.optim.AdamW(
                self.mlm_model.parameters(),
                lr=float(self.pretrain_config.get("learning_rate", 1e-4)),
                weight_decay=float(self.pretrain_config.get("weight_decay", 0.01)),
            )

            mlm_train_loader = DataLoader(
                train_dataset,
                batch_size=mlm_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=lambda b: self.mlm_collate_fn(b, mlm_prob),
            )
            mlm_tuning_loader = DataLoader(
                tuning_dataset,
                batch_size=mlm_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=lambda b: self.mlm_collate_fn(b, mlm_prob),
            )

            best_mlm_loss = float("inf")
            best_mlm_state = None
            no_improve = 0
            for epoch in range(pretrain_epochs):
                self.mlm_model.train()
                total_loss, n_batches = 0.0, 0
                for batch in mlm_train_loader:
                    token_ids = batch["token_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    mlm_optimizer.zero_grad()
                    loss, _ = self.mlm_model(token_ids, attention_mask, labels)
                    loss.backward()
                    mlm_optimizer.step()
                    total_loss += float(loss.item())
                    n_batches += 1
                train_loss = total_loss / max(n_batches, 1)
                val_loss = self._eval_mlm_loss(mlm_tuning_loader, device)
                print(f"[Trial MLM] epoch={epoch + 1} train_loss={train_loss:.4f} tuning_loss={val_loss:.4f}")

                if (best_mlm_loss - val_loss) > pretrain_min_delta:
                    best_mlm_loss = val_loss
                    best_mlm_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= pretrain_patience:
                        print("[Trial MLM] early stopping triggered.")
                        break

            if best_mlm_state is not None:
                self.model.load_state_dict(best_mlm_state)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_config.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 0.01)),
        )
        max_epochs = int(self.training_config.get("epochs", 30))
        patience = int(self.training_config.get("early_stopping_patience", 4))
        min_delta = float(self.training_config.get("early_stopping_min_delta", 1e-4))

        best_tuning_auroc = -np.inf
        best_state = None
        best_epoch = -1
        no_improve = 0

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for batch in train_loader:
                token_ids = batch["token_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                logits = self.model(token_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            tuning_metrics, _, _ = self._eval_classifier(tuning_loader, device)
            tuning_auroc = float(tuning_metrics["auroc"])
            print(f"[Trial CLS] epoch={epoch + 1} train_loss={train_loss:.4f} tuning_auroc={tuning_auroc:.4f}")

            if (tuning_auroc - best_tuning_auroc) > min_delta:
                best_tuning_auroc = tuning_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("[Trial CLS] early stopping triggered.")
                    break

        if best_state is None:
            best_state = copy.deepcopy(self.model.state_dict())
            best_epoch = max_epochs

        self.model.load_state_dict(best_state)
        best_tuning_metrics, tuning_labels, tuning_probs = self._eval_classifier(tuning_loader, device)
        held_out_metrics, held_out_labels, held_out_probs = self._eval_classifier(held_out_loader, device)

        return {
            "trial_cfg": trial_cfg,
            "best_epoch": best_epoch,
            "tuning_metrics": {k: float(v) for k, v in best_tuning_metrics.items()},
            "held_out_metrics": {k: float(v) for k, v in held_out_metrics.items()},
            "model_state": copy.deepcopy(self.model.state_dict()),
            "tuning_labels": tuning_labels,
            "tuning_probs": tuning_probs,
            "held_out_labels": held_out_labels,
            "held_out_probs": held_out_probs,
        }

    def run(self):
        setup_output_dir(self.output_dir)
        self._set_seed(int(self.training_config.get("seed", 42)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        datasets = self._load_datasets()
        filtered = {split: FilteredTokenDataset(ds) for split, ds in datasets.items()}
        self.build_vocabulary(filtered)

        hidden_dims = self.search_config.get("hidden_dims", [256, 512])
        num_layers_values = self.search_config.get("num_layers_values", [4, 6, 8])
        num_heads_values = self.search_config.get("num_heads_values", [4, 8])
        mlm_probs = self.search_config.get("mlm_probabilities", [0.15, 0.20, 0.25])
        selection_metric = self.search_config.get("selection_metric", "auroc")
        if selection_metric != "auroc":
            raise ValueError("Only selection_metric=auroc is currently supported.")

        grid = list(itertools.product(hidden_dims, num_layers_values, num_heads_values, mlm_probs))
        print(f"Running {len(grid)} hyperparameter trials.")

        all_trials = []
        best_trial = None
        best_score = -np.inf
        best_state = None

        for idx, (hd, nl, nh, mp) in enumerate(grid, start=1):
            trial_cfg = {
                "hidden_dim": int(hd),
                "num_layers": int(nl),
                "num_heads": int(nh),
                "mlm_probability": float(mp),
            }
            if trial_cfg["hidden_dim"] % trial_cfg["num_heads"] != 0:
                print(f"Skipping invalid trial {idx}: {trial_cfg} (hidden_dim % num_heads != 0)")
                continue

            print("\n" + "=" * 80)
            print(f"Trial {idx}/{len(grid)}: {trial_cfg}")
            print("=" * 80)
            trial_result = self._run_trial(
                trial_cfg=trial_cfg,
                train_dataset=filtered["train"],
                tuning_dataset=filtered["tuning"],
                held_out_dataset=filtered["held_out"],
                device=device,
            )

            score = float(trial_result["tuning_metrics"]["auroc"])
            all_trials.append(
                {
                    "trial_index": idx,
                    "trial_cfg": trial_cfg,
                    "best_epoch": trial_result["best_epoch"],
                    "tuning_metrics": trial_result["tuning_metrics"],
                    "held_out_metrics": trial_result["held_out_metrics"],
                }
            )
            if score > best_score:
                best_score = score
                best_trial = trial_result
                best_state = copy.deepcopy(trial_result["model_state"])

            save_results({"trials": all_trials}, os.path.join(self.output_dir, "results"), "search_progress.json")

        if best_trial is None:
            raise RuntimeError("No valid hyperparameter trial completed.")

        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        torch.save(best_state, os.path.join(self.output_dir, "models", "best_model.pt"))

        final_model_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        torch.save(best_state, os.path.join(final_model_dir, "pytorch_model.bin"))

        print_results(best_trial["tuning_metrics"], "tuning")
        print_results(best_trial["held_out_metrics"], "held_out")
        plot_all_curves(
            best_trial["tuning_labels"],
            best_trial["tuning_probs"],
            os.path.join(self.output_dir, "plots", "tuning"),
        )
        plot_all_curves(
            best_trial["held_out_labels"],
            best_trial["held_out_probs"],
            os.path.join(self.output_dir, "plots", "held_out"),
        )

        save_results(
            {
                "best_hyperparameters": best_trial["trial_cfg"],
                "best_epoch": int(best_trial["best_epoch"]),
                "best_tuning_metrics": best_trial["tuning_metrics"],
                "final_held_out_metrics": best_trial["held_out_metrics"],
                "num_trials": len(all_trials),
                "all_trials": all_trials,
            },
            self.output_dir,
            "summary.json",
        )
        return best_trial["tuning_metrics"], best_trial["held_out_metrics"]
