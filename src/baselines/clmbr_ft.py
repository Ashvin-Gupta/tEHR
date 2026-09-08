"""Fine-tune CLMBR-T on labeled train examples; pick the checkpoint by tuning AUROC."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.baselines.base import BaseBaseline, setup_output_dir
from src.clmbr.embed import (
    PatientTimeline,
    accumulate_row,
    collate_clmbr_batch,
    events_before,
    finalize_timeline,
    load_model,
    parse_time,
    sequences_to_arrays,
)
from src.clmbr.mapping import CodeMapper
from src.clmbr.model import ClmbrClassifier
from src.clmbr.vocab import ClmbrVocab
from src.evaluations.evaluations import save_results, write_split_artifacts
from src.training.training_utils import init_wandb


def _pred_time(value) -> Optional[datetime]:
    ts = parse_time(value)
    if ts is not None:
        return ts
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return None


class _SeqDataset(Dataset):
    def __init__(self, tokens, ages, y):
        self.tokens = tokens
        self.ages = ages
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tokens[i], self.ages[i], self.y[i]


class CLMBRFineTuneBaseline(BaseBaseline):
    """Supervised fine-tune of frozen-pretrained CLMBR-T + linear head."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        clmbr = config.get("clmbr") or {}
        train_cfg = config.get("training") or {}
        self.model_dir = clmbr.get("model_dir") or config.get("model_dir")
        self.maps_dir = clmbr.get("maps_dir")
        self.meds_dir = (config.get("data") or {}).get("meds_dir")
        self.max_seq = int(clmbr.get("max_seq_len", 496))
        self.overwrite_sequences = bool(clmbr.get("overwrite_sequences", False))
        self.epochs = int(train_cfg.get("epochs", clmbr.get("epochs", 3)))
        self.max_steps = int(train_cfg.get("max_steps", clmbr.get("max_steps", 0)) or 0)
        self.batch_size = int(train_cfg.get("batch_size", clmbr.get("batch_size", 8)))
        self.eval_batch_size = int(
            clmbr.get("eval_batch_size") or train_cfg.get("eval_batch_size") or self.batch_size
        )
        self.lr = float(train_cfg.get("learning_rate", clmbr.get("learning_rate", 1e-5)))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.warmup_steps = int(train_cfg.get("warmup_steps", clmbr.get("warmup_steps", 100)))
        self.eval_every_steps = int(
            clmbr.get("eval_every_steps", train_cfg.get("eval_every_steps", 0)) or 0
        )
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.patience = int(clmbr.get("patience", 3))
        self.seed = int((config.get("seed") or train_cfg.get("seed") or 42))
        self.vocab: Optional[ClmbrVocab] = None
        self.mapper: Optional[CodeMapper] = None
        self.classifier: Optional[ClmbrClassifier] = None
        self.device = None
        if not self.model_dir:
            raise ValueError("clmbr.model_dir is required")
        if not self.meds_dir:
            raise ValueError("data.meds_dir (clean_meds) is required")
        if not self.maps_dir:
            raise ValueError("clmbr.maps_dir is required")

    def setup_wandb(self) -> str:
        return init_wandb(
            self.config, local_rank=self.local_rank, default_project="tEHR-baselines"
        )[0]

    def load_model(self):
        print(f"Loading CLMBR-T from {self.model_dir}")
        self.vocab = ClmbrVocab.from_pretrained(self.model_dir)
        self.mapper = CodeMapper.load(self.maps_dir, self.vocab)
        backbone, self.device = load_model(self.model_dir)
        backbone.train()
        self.classifier = ClmbrClassifier(backbone).to(self.device)
        print(f"  vocab={self.vocab.vocab_size:,} device={self.device}")

    def _load_labels(self, split: str) -> List[dict]:
        aces_dir = self.data_config.get("aces_label_dir")
        path = os.path.join(aces_dir, f"{split}_labels.parquet")
        df = pl.read_parquet(path)
        label_col = self.data_config.get("label_column", "label")
        rows = []
        for rec in df.iter_rows(named=True):
            sid = rec.get("subject_id")
            if sid is None:
                continue
            rows.append(
                {
                    "subject_id": int(sid),
                    "prediction_time": rec.get("prediction_time"),
                    "label": int(rec.get(label_col, 0) or 0),
                }
            )
        return rows

    def _extract_split(self, split: str) -> Tuple[list, list, np.ndarray]:
        assert self.vocab and self.mapper
        labels = self._load_labels(split)
        needed = defaultdict(list)
        for i, row in enumerate(labels):
            needed[row["subject_id"]].append(i)

        shard_dir = os.path.join(self.meds_dir, split)
        files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".parquet"))
        timelines: dict[int, PatientTimeline] = {}
        print(f"Scanning {split} MEDS ({len(files)} shards, {len(labels):,} labels)…")
        for fname in tqdm(files, desc=f"{split} shards"):
            df = pl.read_parquet(
                os.path.join(shard_dir, fname),
                columns=["subject_id", "time", "code", "numeric_value"],
            )
            for rec in df.iter_rows(named=True):
                sid = rec.get("subject_id")
                if sid is None or int(sid) not in needed:
                    continue
                sid = int(sid)
                if sid not in timelines:
                    timelines[sid] = PatientTimeline()
                accumulate_row(
                    timelines[sid],
                    rec.get("code"),
                    rec.get("time"),
                    rec.get("numeric_value"),
                    self.mapper,
                    self.vocab,
                )
        for sid in list(timelines):
            timelines[sid] = finalize_timeline(timelines[sid], self.vocab)

        sequences = []
        y = np.zeros(len(labels), dtype=np.int64)
        empty = 0
        for i, row in enumerate(labels):
            y[i] = row["label"]
            pred = _pred_time(row["prediction_time"])
            tl = timelines.get(row["subject_id"])
            if tl is None or pred is None:
                sequences.append([])
                empty += 1
                continue
            seq = events_before(tl, pred, max_len=self.max_seq)
            if not seq:
                empty += 1
            sequences.append(seq)
        tokens, ages = sequences_to_arrays(sequences)
        n_tok = np.array([0 if not s else len(s) for s in sequences], dtype=np.int32)
        print(
            f"  {split}: {len(sequences):,} examples empty={empty} "
            f"mean tokens={n_tok.mean():.1f} pos={y.mean():.4f}"
        )
        return tokens, ages, y

    def _cached_split(self, split: str):
        cache_dir = os.path.join(self.output_dir, "sequences")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{split}.npz")
        if os.path.isfile(path) and not self.overwrite_sequences:
            data = np.load(path, allow_pickle=True)
            print(f"Loaded cached sequences {path} n={len(data['y'])}")
            return list(data["tokens"]), list(data["ages"]), data["y"]
        tokens, ages, y = self._extract_split(split)
        np.savez_compressed(path, tokens=np.array(tokens, dtype=object), ages=np.array(ages, dtype=object), y=y)
        return tokens, ages, y

    def _loader(self, tokens, ages, y, train: bool) -> DataLoader:
        ds = _SeqDataset(tokens, ages, y)
        return DataLoader(
            ds,
            batch_size=self.batch_size if train else self.eval_batch_size,
            shuffle=train,
            drop_last=False,
            collate_fn=lambda rows: collate_clmbr_batch(
                rows, torch.device("cpu"), self.max_seq
            ),
        )

    def _lr_at(self, step: int, total_steps: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.lr * float(step + 1) / float(self.warmup_steps)
        remain = max(total_steps - self.warmup_steps, 1)
        progress = min(max(step - self.warmup_steps, 0) / remain, 1.0)
        return self.lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    @torch.no_grad()
    def _predict_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        assert self.classifier
        self.classifier.eval()
        probs, labels = [], []
        for tokens, ages, mask, lengths, y in loader:
            tokens = tokens.to(self.device, non_blocking=True)
            ages = ages.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            logits = self.classifier(tokens, ages, mask, lengths)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(y.cpu().numpy())
        self.classifier.train()
        return np.concatenate(labels), np.concatenate(probs)

    def _score_split(self, tokens, ages, y, split: str) -> Dict[str, float]:
        loader = self._loader(tokens, ages, y, train=False)
        labels, probs = self._predict_loader(loader)
        return write_split_artifacts(self.output_dir, split, labels, probs)

    def _train(self, train_pack, val_pack) -> Dict[str, Any]:
        assert self.classifier
        tokens_t, ages_t, y_t = train_pack
        tokens_v, ages_v, y_v = val_pack
        loader = self._loader(tokens_t, ages_t, y_t, train=True)
        steps_per_epoch = max(len(loader), 1)
        total_steps = self.epochs * steps_per_epoch
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)
        n_pos = max(int((y_t == 1).sum()), 1)
        n_neg = max(int((y_t == 0).sum()), 1)
        pos_weight = torch.tensor(n_neg / n_pos, device=self.device)
        print(
            f"Fine-tune: epochs={self.epochs} max_steps={total_steps} "
            f"batch={self.batch_size} lr={self.lr:g} pos_weight={float(pos_weight):.2f}"
        )

        opt = torch.optim.AdamW(
            self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        best_auc, best_state, bad = -1.0, None, 0
        history = []
        step = 0
        self.classifier.train()
        for epoch in range(self.epochs):
            running = 0.0
            n_seen = 0
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{self.epochs}")
            for tokens, ages, mask, lengths, y in pbar:
                if step >= total_steps:
                    break
                tokens = tokens.to(self.device, non_blocking=True)
                ages = ages.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                lengths = lengths.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                lr = self._lr_at(step, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.zero_grad(set_to_none=True)
                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else torch.autocast(device_type="cpu", enabled=False)
                )
                with ctx:
                    logits = self.classifier(tokens, ages, mask, lengths)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=pos_weight
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                opt.step()
                running += float(loss.detach()) * len(y)
                n_seen += len(y)
                step += 1
                pbar.set_postfix(loss=f"{float(loss):.4f}", lr=f"{lr:.2e}")

                do_eval = step == total_steps or (step % steps_per_epoch == 0)
                if self.eval_every_steps > 0:
                    do_eval = do_eval or (step % self.eval_every_steps == 0)
                if do_eval:
                    labels, probs = self._predict_loader(self._loader(tokens_v, ages_v, y_v, False))
                    try:
                        auc = float(roc_auc_score(labels, probs))
                    except ValueError:
                        auc = 0.0
                    row = {
                        "step": step,
                        "epoch": epoch + 1,
                        "train_loss": running / max(n_seen, 1),
                        "tuning_auroc": auc,
                        "lr": lr,
                    }
                    history.append(row)
                    print(
                        f"  step {step}/{total_steps}  train_loss={row['train_loss']:.4f}  "
                        f"tuning AUROC={auc:.4f}"
                    )
                    if auc > best_auc:
                        best_auc = auc
                        bad = 0
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.classifier.state_dict().items()
                        }
                        print(f"  new best tuning AUROC={best_auc:.4f}")
                    else:
                        bad += 1
                        if bad >= self.patience:
                            print(f"  early stop at step {step} (patience={self.patience})")
                            step = total_steps
                            break
            if step >= total_steps:
                break

        if best_state is not None:
            self.classifier.load_state_dict(best_state)
        return {
            "best_tuning_auroc": best_auc,
            "steps": step,
            "pos_weight": float(pos_weight),
            "history": history,
        }

    def save_model(self):
        assert self.classifier
        path = os.path.join(self.output_dir, "models", "clmbr_ft.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.classifier.state_dict(),
                "lr": self.lr,
                "epochs": self.epochs,
                "max_steps": self.max_steps,
                "batch_size": self.batch_size,
            },
            path,
        )
        print(f"  saved {path}")

    def run(self, resume_from_checkpoint: str | None = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.setup_wandb()
        setup_output_dir(self.output_dir, overwrite=True)
        self.load_model()
        train_pack = self._cached_split("train")
        val_pack = self._cached_split("tuning")
        test_pack = self._cached_split("held_out")
        train_info = self._train(train_pack, val_pack)
        self.save_model()
        val_metrics = self._score_split(*val_pack, "tuning")
        test_metrics = self._score_split(*test_pack, "held_out")
        save_results(
            {
                "validation": {k: float(v) for k, v in val_metrics.items()},
                "test": {k: float(v) for k, v in test_metrics.items()},
                "fine_tune": {
                    "best_tuning_auroc": train_info["best_tuning_auroc"],
                    "steps": train_info["steps"],
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "max_steps": self.max_steps,
                    "batch_size": self.batch_size,
                    "pos_weight": train_info["pos_weight"],
                    "history": train_info["history"],
                },
                "n_train": int(len(train_pack[2])),
                "n_tuning": int(len(val_pack[2])),
                "n_held_out": int(len(test_pack[2])),
            },
            self.output_dir,
            "summary.json",
        )
        return val_metrics, test_metrics

    def run_eval_only(self, checkpoint_path: str):
        from src.training.training_utils import resolve_weight_path

        self.load_model()
        bundle = torch.load(resolve_weight_path(checkpoint_path), map_location=self.device)
        self.classifier.load_state_dict(bundle["state_dict"])
        val_pack = self._cached_split("tuning")
        test_pack = self._cached_split("held_out")
        return self._score_split(*val_pack, "tuning"), self._score_split(*test_pack, "held_out")

    def predict(self, dataset) -> np.ndarray:
        raise NotImplementedError("CLMBRFineTuneBaseline uses cached sequences, not Dataset.predict")
