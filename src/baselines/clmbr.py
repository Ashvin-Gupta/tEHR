"""Frozen CLMBR-T-base embeddings + logistic regression probe."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.baselines.base import BaseBaseline, setup_output_dir
from src.clmbr.embed import (
    PatientTimeline,
    accumulate_row,
    embed_sequences,
    events_before,
    finalize_timeline,
    load_model,
    parse_time,
)
from src.clmbr.mapping import CodeMapper
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


class CLMBRBaseline(BaseBaseline):
    """EHRSHOT-style linear probe on frozen CLMBR-T representations."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        clmbr = config.get("clmbr") or {}
        self.model_dir = clmbr.get("model_dir") or config.get("model_dir")
        self.maps_dir = clmbr.get("maps_dir")
        self.meds_dir = (config.get("data") or {}).get("meds_dir")
        self.max_seq = int(clmbr.get("max_seq_len", 496))
        self.C_grid = clmbr.get("C_grid") or [1e-3, 1e-2, 1e-1, 1.0, 10.0]
        self.overwrite_embeddings = bool(clmbr.get("overwrite_embeddings", False))
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[LogisticRegression] = None
        self.vocab: Optional[ClmbrVocab] = None
        self.mapper: Optional[CodeMapper] = None
        self.transformer = None
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
        print(f"Loading CLMBR vocab from {self.model_dir}")
        self.vocab = ClmbrVocab.from_pretrained(self.model_dir)
        self.mapper = CodeMapper.load(self.maps_dir, self.vocab)
        self.transformer, self.device = load_model(self.model_dir)
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
            lab = rec.get(label_col, 0)
            rows.append(
                {
                    "subject_id": int(sid),
                    "prediction_time": rec.get("prediction_time"),
                    "label": int(lab or 0),
                }
            )
        return rows

    def _extract_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.vocab and self.mapper and self.transformer
        labels = self._load_labels(split)
        needed = defaultdict(list)
        for i, row in enumerate(labels):
            needed[row["subject_id"]].append(i)

        shard_dir = os.path.join(self.meds_dir, split)
        files = sorted(
            f
            for f in os.listdir(shard_dir)
            if f.endswith(".parquet")
        )
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

        print(f"  embedding {len(sequences):,} examples ({empty:,} empty) on {self.device}")
        X = embed_sequences(sequences, self.transformer, self.vocab, self.device)
        n_tok = np.array([len(s) for s in sequences], dtype=np.int32)
        print(
            f"  mean mapped tokens/example={n_tok.mean():.1f} "
            f"median={np.median(n_tok):.0f} empty={empty}"
        )
        return X, y, n_tok

    def _fit_probe(self, X_train, y_train, X_val, y_val) -> LogisticRegression:
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)
        best_clf, best_auc, best_C = None, -1.0, None
        from sklearn.metrics import roc_auc_score

        for C in self.C_grid:
            clf = LogisticRegression(
                C=float(C),
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
            )
            clf.fit(Xt, y_train)
            probs = clf.predict_proba(Xv)[:, 1]
            try:
                auc = float(roc_auc_score(y_val, probs))
            except ValueError:
                auc = 0.0
            print(f"  C={C:g}  tuning AUROC={auc:.4f}")
            if auc > best_auc:
                best_auc, best_clf, best_C = auc, clf, C
        print(f"Selected C={best_C} (tuning AUROC={best_auc:.4f})")
        self.scaler = scaler
        self.clf = best_clf
        return best_clf

    def save_model(self):
        import joblib

        path = os.path.join(self.output_dir, "models", "clmbr_probe.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf, "C_grid": self.C_grid}, path)
        print(f"  saved {path}")

    def run(self, resume_from_checkpoint: str | None = None):
        self.setup_wandb()
        setup_output_dir(self.output_dir, overwrite=True)
        self.load_model()

        cache_dir = os.path.join(self.output_dir, "embeddings")
        os.makedirs(cache_dir, exist_ok=True)

        def cached(split):
            npz = os.path.join(cache_dir, f"{split}.npz")
            if os.path.isfile(npz) and not self.overwrite_embeddings:
                data = np.load(npz)
                print(f"Loaded cached embeddings {npz} {data['X'].shape}")
                return data["X"], data["y"], data["n_tokens"]
            X, y, n_tok = self._extract_split(split)
            np.savez_compressed(npz, X=X, y=y, n_tokens=n_tok)
            return X, y, n_tok

        X_train, y_train, _ = cached("train")
        X_val, y_val, _ = cached("tuning")
        X_test, y_test, _ = cached("held_out")

        self._fit_probe(X_train, y_train, X_val, y_val)
        self.save_model()

        def score(X, y, split):
            probs = self.clf.predict_proba(self.scaler.transform(X))[:, 1]
            return write_split_artifacts(self.output_dir, split, y, probs)

        val_metrics = score(X_val, y_val, "tuning")
        test_metrics = score(X_test, y_test, "held_out")
        save_results(
            {
                "validation": {k: float(v) for k, v in val_metrics.items()},
                "test": {k: float(v) for k, v in test_metrics.items()},
                "C": float(self.clf.C),
                "n_train": int(len(y_train)),
                "n_tuning": int(len(y_val)),
                "n_held_out": int(len(y_test)),
            },
            self.output_dir,
            "summary.json",
        )
        return val_metrics, test_metrics

    def run_eval_only(self, checkpoint_path: str):
        import joblib

        from src.training.training_utils import resolve_weight_path

        bundle = joblib.load(resolve_weight_path(checkpoint_path))
        self.scaler, self.clf = bundle["scaler"], bundle["clf"]
        self.load_model()
        X_val, y_val, _ = self._extract_split("tuning")
        X_test, y_test, _ = self._extract_split("held_out")
        val_metrics = write_split_artifacts(
            self.output_dir,
            "tuning",
            y_val,
            self.clf.predict_proba(self.scaler.transform(X_val))[:, 1],
        )
        test_metrics = write_split_artifacts(
            self.output_dir,
            "held_out",
            y_test,
            self.clf.predict_proba(self.scaler.transform(X_test))[:, 1],
        )
        return val_metrics, test_metrics

    def predict(self, dataset) -> np.ndarray:
        raise NotImplementedError("CLMBRBaseline uses extract_split, not Dataset.predict")
