"""Resume and shard long evals so they fit a 24h Slurm wall.

Jobs can:

- flush ``{split}_partial.json`` every ``EVAL_FLUSH_EVERY`` samples (default 250)
  and pick up from there on the next submit
- split the test set with ``EVAL_SHARD=i/n`` (e.g. ``0/2`` and ``1/2``) and
  merge once every shard file is on disk
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def flush_every() -> int:
    raw = os.environ.get("EVAL_FLUSH_EVERY", "250").strip()
    try:
        n = int(raw)
    except ValueError:
        return 250
    return max(1, n)


def parse_eval_shard(raw: str | None = None) -> tuple[int, int] | None:
    """Parse ``EVAL_SHARD=i/n``. ``i`` is 0-based; ``n`` must be >= 2."""
    text = raw if raw is not None else os.environ.get("EVAL_SHARD", "").strip()
    if not text:
        return None
    try:
        i_s, n_s = text.split("/", 1)
        index, n_shards = int(i_s), int(n_s)
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: EVAL_SHARD={text!r} must look like 0/2"
        ) from exc
    if n_shards < 2 or index < 0 or index >= n_shards:
        raise SystemExit(
            f"ERROR: EVAL_SHARD={text!r} needs 0 <= i < n and n >= 2"
        )
    return index, n_shards


def shard_bounds(n_items: int, index: int, n_shards: int) -> tuple[int, int]:
    start = (n_items * index) // n_shards
    end = (n_items * (index + 1)) // n_shards
    return start, end


def partial_pred_path(output_dir: str, tag: str) -> str:
    return os.path.join(output_dir, "results", f"{tag}_partial.json")


def shard_result_path(output_dir: str, split_name: str, index: int, n_shards: int) -> str:
    return os.path.join(
        output_dir, "results", f"{split_name}_shard_{index}of{n_shards}.json"
    )


def load_eval_partial(path: str, n_expected: int) -> list[float]:
    if not os.path.isfile(path):
        return []
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  Ignoring unreadable partial {path}: {exc}", flush=True)
        return []
    if int(data.get("n_expected", -1)) != int(n_expected):
        print(
            f"  Ignoring partial {path}: n_expected "
            f"{data.get('n_expected')} != {n_expected}",
            flush=True,
        )
        return []
    probs = data.get("probs") or []
    if not isinstance(probs, list) or not probs:
        return []
    print(
        f"  Resuming {path}: {len(probs)}/{n_expected} already scored",
        flush=True,
    )
    return [float(p) for p in probs]


def save_eval_partial(path: str, probs: list[float], n_expected: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    payload = {
        "n_expected": int(n_expected),
        "n_done": len(probs),
        "probs": probs,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def maybe_flush_partial(
    path: str | None,
    probs: list[float],
    n_expected: int,
    *,
    force: bool = False,
) -> None:
    if not path:
        return
    if not force and len(probs) % flush_every() != 0:
        return
    save_eval_partial(path, probs, n_expected)


def write_shard_file(
    path: str,
    *,
    split_name: str,
    index: int,
    n_shards: int,
    start: int,
    end: int,
    n_total: int,
    labels: list[Any],
    probs: list[float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "split": split_name,
        "shard_index": index,
        "n_shards": n_shards,
        "start": start,
        "end": end,
        "n_total": n_total,
        "labels": labels,
        "probs": probs,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    print(f"  - Shard saved to {path} ({end - start} of {n_total})", flush=True)


def try_merge_shards(
    output_dir: str,
    split_name: str,
    n_shards: int,
    n_total: int,
) -> tuple[list[Any], list[float]] | None:
    parts: list[dict[str, Any]] = []
    for index in range(n_shards):
        path = shard_result_path(output_dir, split_name, index, n_shards)
        if not os.path.isfile(path):
            print(f"  Shard {index}/{n_shards} not on disk yet ({path})", flush=True)
            return None
        parts.append(json.loads(Path(path).read_text()))

    labels: list[Any] = []
    probs: list[float] = []
    cursor = 0
    for part in parts:
        start = int(part["start"])
        end = int(part["end"])
        if start != cursor or int(part.get("n_total", -1)) != n_total:
            print(
                f"  Cannot merge shards: expected start={cursor}, "
                f"got start={start} n_total={part.get('n_total')}",
                flush=True,
            )
            return None
        labels.extend(part.get("labels") or [])
        probs.extend(float(p) for p in (part.get("probs") or []))
        cursor = end
    if cursor != n_total or len(labels) != n_total or len(probs) != n_total:
        print(
            f"  Cannot merge shards: concatenated {len(labels)} labels / "
            f"{len(probs)} probs, expected {n_total}",
            flush=True,
        )
        return None
    print(f"  Merged {n_shards} shards → {n_total} test predictions", flush=True)
    return labels, probs
