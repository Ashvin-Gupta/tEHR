import copy
import itertools
import os
import random
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.baseline_encoder_transformer import (
    EncoderTransformerBaseline,
    FilteredTokenDataset,
)
from src.baselines.baseline_utils import setup_output_dir
from src.data.dataset_factory import get_dataset
from src.evaluations.evaluations import (
    compute_baseline_metrics,
    plot_all_curves,
    print_results,
    save_results,
)


class EncoderTransformerHyperparamSearchBaseline:
    """Grid search wrapper around EncoderTransformerBaseline."""

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
        return {split: get_dataset(dataset_name, data_cfg, split) for split in ["train", "tuning", "held_out"]}

    def _run_trial(
        self,
        trial_cfg: Dict[str, Any],
        train_dataset: Any,
        tuning_dataset: Any,
        held_out_dataset: Any,
        device: torch.device,
        shared_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        trial_config = copy.deepcopy(self.config)
        trial_config.setdefault("model", {})
        trial_config.setdefault("pretrain", {})
        trial_config["model"]["hidden_dim"] = int(trial_cfg["hidden_dim"])
        trial_config["model"]["num_layers"] = int(trial_cfg["num_layers"])
        trial_config["model"]["num_heads"] = int(trial_cfg["num_heads"])
        trial_config["pretrain"]["mlm_probability"] = float(trial_cfg["mlm_probability"])

        baseline = EncoderTransformerBaseline(trial_config)
        baseline.output_dir = self.output_dir

        # Reuse one-time prepared vocabulary/tokenization state for all trials.
        baseline.vocab = shared_state["vocab"]
        baseline.token_to_id = shared_state["token_to_id"]
        baseline.id_to_token = shared_state["id_to_token"]
        baseline._label_conversion_warned = True

        baseline.create_model()
        baseline.model = baseline.model.to(device)

        batch_size = int(self.training_config.get("batch_size", 32))
        num_workers = int(self.training_config.get("dataloader_num_workers", 2))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=baseline.collate_fn,
            num_workers=num_workers,
        )
        tuning_loader = DataLoader(
            tuning_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=baseline.collate_fn,
            num_workers=num_workers,
        )
        held_out_loader = DataLoader(
            held_out_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=baseline.collate_fn,
            num_workers=num_workers,
        )

        if self.pretrain_config.get("enabled", True):
            mlm_batch_size = int(self.pretrain_config.get("batch_size", batch_size))
            pretrain_epochs = int(self.pretrain_config.get("epochs", 20))
            pretrain_patience = int(self.pretrain_config.get("early_stopping_patience", 3))
            pretrain_min_delta = float(self.pretrain_config.get("early_stopping_min_delta", 1e-4))

            baseline.create_mlm_model()
            baseline.mlm_model = baseline.mlm_model.to(device)
            mlm_optimizer = torch.optim.AdamW(
                baseline.mlm_model.parameters(),
                lr=float(self.pretrain_config.get("learning_rate", 1e-4)),
                weight_decay=float(self.pretrain_config.get("weight_decay", 0.01)),
            )

            mlm_train_loader = DataLoader(
                train_dataset,
                batch_size=mlm_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=partial(baseline.mlm_collate_fn),
            )
            mlm_tuning_loader = DataLoader(
                tuning_dataset,
                batch_size=mlm_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=partial(baseline.mlm_collate_fn),
            )

            best_mlm_loss = float("inf")
            best_mlm_state = None
            no_improve = 0
            for epoch in range(pretrain_epochs):
                train_loss = baseline.train_mlm_epoch(mlm_train_loader, mlm_optimizer, device)
                val_loss = baseline.evaluate_mlm(mlm_tuning_loader, device)
                print(f"[Trial MLM] epoch={epoch + 1} train_loss={train_loss:.4f} tuning_loss={val_loss:.4f}")

                if (best_mlm_loss - val_loss) > pretrain_min_delta:
                    best_mlm_loss = val_loss
                    best_mlm_state = copy.deepcopy(baseline.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= pretrain_patience:
                        print("[Trial MLM] early stopping triggered.")
                        break

            if best_mlm_state is not None:
                baseline.model.load_state_dict(best_mlm_state)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            baseline.model.parameters(),
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

        def eval_metrics_only(data_loader: DataLoader) -> Dict[str, float]:
            baseline.model.eval()
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch in data_loader:
                    token_ids = batch["token_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    logits = baseline.model(token_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())
            return compute_baseline_metrics(np.array(all_labels), np.array(all_probs))

        for epoch in range(max_epochs):
            train_loss = baseline.train_epoch(train_loader, optimizer, criterion, device)
            tuning_metrics = eval_metrics_only(tuning_loader)
            tuning_auroc = float(tuning_metrics["auroc"])
            print(f"[Trial CLS] epoch={epoch + 1} train_loss={train_loss:.4f} tuning_auroc={tuning_auroc:.4f}")

            if (tuning_auroc - best_tuning_auroc) > min_delta:
                best_tuning_auroc = tuning_auroc
                best_state = copy.deepcopy(baseline.model.state_dict())
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("[Trial CLS] early stopping triggered.")
                    break

        if best_state is None:
            best_state = copy.deepcopy(baseline.model.state_dict())
            best_epoch = max_epochs

        baseline.model.load_state_dict(best_state)
        best_tuning_metrics = baseline.evaluate(tuning_loader, device, "tuning")
        held_out_metrics = baseline.evaluate(held_out_loader, device, "held_out")

        tuning_results = os.path.join(self.output_dir, "results", "tuning_results.json")
        held_out_results = os.path.join(self.output_dir, "results", "held_out_results.json")
        tuning_labels = np.array([])
        tuning_probs = np.array([])
        held_out_labels = np.array([])
        held_out_probs = np.array([])
        if os.path.exists(tuning_results):
            import json

            with open(tuning_results, "r") as f:
                data = json.load(f)
                tuning_labels = np.array(data.get("labels", []))
                tuning_probs = np.array(data.get("probs", []))
        if os.path.exists(held_out_results):
            import json

            with open(held_out_results, "r") as f:
                data = json.load(f)
                held_out_labels = np.array(data.get("labels", []))
                held_out_probs = np.array(data.get("probs", []))

        return {
            "trial_cfg": trial_cfg,
            "best_epoch": best_epoch,
            "tuning_metrics": {k: float(v) for k, v in best_tuning_metrics.items()},
            "held_out_metrics": {k: float(v) for k, v in held_out_metrics.items()},
            "model_state": copy.deepcopy(baseline.model.state_dict()),
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

        # Build vocabulary exactly once and reuse it in every trial.
        shared = EncoderTransformerBaseline(copy.deepcopy(self.config))
        shared.output_dir = self.output_dir
        shared.build_vocabulary(filtered)
        shared_state = {
            "vocab": shared.vocab,
            "token_to_id": shared.token_to_id,
            "id_to_token": shared.id_to_token,
        }

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
                shared_state=shared_state,
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
