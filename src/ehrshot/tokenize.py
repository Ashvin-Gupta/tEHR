"""Build tEHR pickle shards + ACES-style labels from EHRSHOT_ASSETS.

Does **not** use EHRSHOT_massaging (that path capped histories at 200 events
for an old LLM-prompt dump). This writes full timelines, one trajectory per
patient, with prediction-time cutoff applied later in EHRSHOTDataset.

    python -m src.ehrshot.tokenize

Outputs:
  {output_dir}/{train,tuning,held_out}/*.pkl
  {output_dir}/vocab.csv
  {labels_out}/{task}/{train,tuning,held_out}_labels.parquet
"""
from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import polars as pl

SPLIT_MAP = {"train": "train", "val": "tuning", "test": "held_out"}
TASKS = {
    "icu_admission": "guo_icu",
    "readmission_30d": "guo_readmission",
    "pancreatic_cancer": "new_pancan",
}
SPECIAL = ("<unknown>", "<start>", "<end>")
PERSON_PREFIXES = ("Gender/", "Race/", "Ethnicity/", "Birth")
RACE_NAME = {
    "0": "UNKNOWN",
    "1": "AMERICAN_INDIAN",
    "2": "ASIAN",
    "3": "BLACK",
    "4": "WHITE",
    "5": "PACIFIC_ISLANDER",
}

_PROJECT = os.environ.get("PROJECTDIR") or "/projects/u6ps"
DEFAULT_ASSETS = os.path.join(_PROJECT, "EHRSHOT_ASSETS")
DEFAULT_OUT = os.path.join(_PROJECT, "EHRSHOT_tokenisation")
DEFAULT_LABELS = os.path.join(_PROJECT, "EHRSHOT_data/labels")


def _group_pid(key) -> int:
    if isinstance(key, tuple):
        key = key[0]
    return int(key)


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None
    text = text.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _time_gap_token(prev: datetime, curr: datetime) -> str | None:
    delta = curr - prev
    seconds = delta.total_seconds()
    if seconds < 3600:
        return None
    hours = int(seconds // 3600)
    if hours < 24:
        unit = "hour" if hours == 1 else "hours"
        return f"TIME {hours} {unit}"
    days = int(seconds // 86400)
    if days < 7:
        unit = "day" if days == 1 else "days"
        return f"TIME {days} {unit}"
    weeks = max(days // 7, 1)
    unit = "week" if weeks == 1 else "weeks"
    return f"TIME {weeks} {unit}"


def _value_token(raw) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"none", "nan", ""}:
        return None
    try:
        return f"{float(text):.1f}"
    except ValueError:
        return f"VALUE//{text}"


def _unit_token(raw) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().replace("[", "").replace("]", "")
    if not text or text.lower() in {"none", "nan"}:
        return None
    return f"UNIT//{text}"


def _event_token(table: str, code: str) -> str:
    table = (table or "unknown").strip() or "unknown"
    code = (code or "UNKNOWN").strip() or "UNKNOWN"
    return f"EVENT//{table}//{code}"


def _person_demographics(rows: list[dict]) -> dict[str, Any]:
    """Gender / race / DOB live on omop_table=person (not Unknown stubs)."""
    gender = None
    race = None
    ethnicity = None
    dob = None
    for rec in rows:
        code = str(rec.get("code") or "")
        ts = rec.get("start")
        if ts is not None and (dob is None or ts < dob):
            dob = ts
        if code.startswith("Gender/"):
            gender = code.split("/", 1)[1].upper() or None
        elif code.startswith("Race/"):
            key = code.split("/", 1)[1]
            race = RACE_NAME.get(key, key.upper() or None)
        elif code.startswith("Ethnicity/"):
            ethnicity = code.split("/", 1)[1].upper().replace(" ", "_") or None
        elif code == "Birth" or code.startswith("Birth"):
            dob = ts or dob
    return {"gender": gender, "race": race, "ethnicity": ethnicity, "dob": dob}


def _age_years(dob: datetime | None, at: datetime | None) -> int | None:
    if dob is None or at is None or at < dob:
        return None
    return int((at - dob).days / 365.25)


def tokenize_patient(
    clinical: list[dict],
    demo: dict[str, Any],
) -> tuple[list[str], list[datetime]]:
    clinical = sorted(clinical, key=lambda r: (r["start"] or datetime.min, str(r.get("code") or "")))
    first_t = clinical[0]["start"] if clinical else demo.get("dob")
    tokens: list[str] = ["<start>"]
    times: list[datetime] = [first_t or datetime.min]

    age = _age_years(demo.get("dob"), first_t)
    if age is not None:
        tokens.extend(["AGE", str(age)])
        times.extend([first_t, first_t])
    if demo.get("gender"):
        tokens.append(f"GENDER//{demo['gender']}")
        times.append(first_t)
    if demo.get("race"):
        tokens.append(f"RACE//{demo['race']}")
        times.append(first_t)
    if demo.get("ethnicity"):
        tokens.append(f"ETHNICITY//{demo['ethnicity']}")
        times.append(first_t)

    prev = None
    for rec in clinical:
        ts = rec["start"]
        if ts is None:
            continue
        if prev is not None:
            gap = _time_gap_token(prev, ts)
            if gap:
                tokens.append(gap)
                times.append(ts)
        prev = ts
        tokens.append(_event_token(rec.get("omop_table"), rec.get("code")))
        times.append(ts)
        val = _value_token(rec.get("value"))
        if val:
            tokens.append(val)
            times.append(ts)
            unit = _unit_token(rec.get("unit"))
            if unit:
                tokens.append(unit)
                times.append(ts)
    return tokens, times


def _write_shards(records: list[dict], split_dir: str, shard_size: int = 200) -> int:
    os.makedirs(split_dir, exist_ok=True)
    n = 0
    for i in range(0, len(records), shard_size):
        chunk = records[i : i + shard_size]
        path = os.path.join(split_dir, f"shard_{n:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        n += 1
    return n


def write_vocab(counts: Counter[str], path: str) -> dict[str, int]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    token_to_id: dict[str, int] = {}
    rows = []
    for i, spec in enumerate(SPECIAL):
        token_to_id[spec] = i
        rows.append((i, spec, int(counts.get(spec, 0))))
    next_id = len(SPECIAL)
    for tok, n in counts.most_common():
        if tok in token_to_id:
            continue
        token_to_id[tok] = next_id
        rows.append((next_id, tok, int(n)))
        next_id += 1
    with open(path, "w") as f:
        f.write("token,str,count\n")
        for i, tok, n in rows:
            safe = str(tok).replace('"', '""')
            if any(c in safe for c in ",\n"):
                safe = f'"{safe}"'
            f.write(f"{i},{safe},{n}\n")
    return token_to_id


def write_task_labels(
    labels_dir: str,
    splits_df: pl.DataFrame,
    out_root: str,
) -> None:
    split_lookup = {
        int(r["omop_person_id"]): SPLIT_MAP[str(r["split"])]
        for r in splits_df.iter_rows(named=True)
        if str(r["split"]) in SPLIT_MAP
    }
    for task, folder in TASKS.items():
        src = os.path.join(labels_dir, folder, "labeled_patients.csv")
        raw = pl.read_csv(src)
        rows_by_split: dict[str, list[dict]] = defaultdict(list)
        n_skip = 0
        for rec in raw.iter_rows(named=True):
            pid = int(rec["patient_id"])
            split = split_lookup.get(pid)
            if split is None:
                n_skip += 1
                continue
            value = rec.get("value")
            if isinstance(value, str):
                label = 1 if value.strip().lower() in {"true", "1", "yes"} else 0
            else:
                label = 1 if bool(value) else 0
            rows_by_split[split].append(
                {
                    "subject_id": pid,
                    "prediction_time": rec["prediction_time"],
                    "label": label,
                }
            )
        task_dir = os.path.join(out_root, task)
        os.makedirs(task_dir, exist_ok=True)
        for split in ("train", "tuning", "held_out"):
            part = pl.DataFrame(rows_by_split[split]) if rows_by_split[split] else pl.DataFrame(
                schema={"subject_id": pl.Int64, "prediction_time": pl.Utf8, "label": pl.Int64}
            )
            if "prediction_time" in part.columns and part.schema["prediction_time"] == pl.Utf8:
                part = part.with_columns(
                    pl.col("prediction_time").str.to_datetime(strict=False)
                )
            dest = os.path.join(task_dir, f"{split}_labels.parquet")
            part.write_parquet(dest)
            pos = int(part["label"].sum()) if len(part) else 0
            print(f"  {task} {split}: {len(part):,} labels pos={pos} → {dest}")
        if n_skip:
            print(f"  {task}: skipped {n_skip} rows with no split")


def _csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """Stream the 3GB CSV to parquet so later patient filters stay small."""
    if os.path.isfile(parquet_path):
        print(f"  reusing {parquet_path}")
        return
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    print(f"  streaming {csv_path} → {parquet_path}", flush=True)
    try:
        pl.scan_csv(csv_path, infer_schema_length=10000).sink_parquet(
            parquet_path, compression="zstd"
        )
        return
    except Exception as exc:
        print(f"  polars sink_parquet failed ({exc}); using pyarrow", flush=True)
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    reader = pacsv.open_csv(csv_path)
    writer = None
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()


def _with_start_datetime(lf: pl.LazyFrame) -> pl.LazyFrame:
    dtype = lf.collect_schema()["start"]
    if dtype == pl.Utf8:
        return lf.with_columns(pl.col("start").str.to_datetime(strict=False))
    return lf


def _scan_events(path: str) -> pl.LazyFrame:
    if path.endswith(".parquet"):
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path, infer_schema_length=10000)
    return _with_start_datetime(lf)


def _iter_patient_chunks(events_lf: pl.LazyFrame, pids: list[int], chunk_size: int = 250):
    lf = events_lf.filter(pl.col("omop_table") != "person")
    for i in range(0, len(pids), chunk_size):
        chunk = pids[i : i + chunk_size]
        yield lf.filter(pl.col("patient_id").is_in(chunk)).collect()


def _row_start(rec: dict) -> datetime | None:
    start = rec.get("start")
    if isinstance(start, datetime):
        return start
    return _parse_time(start)


def build(args: argparse.Namespace) -> None:
    print(f"Loading splits from {args.splits_csv}")
    splits = pl.read_csv(args.splits_csv)
    pid_to_split = {
        int(r["omop_person_id"]): SPLIT_MAP[str(r["split"])]
        for r in splits.iter_rows(named=True)
        if str(r["split"]) in SPLIT_MAP
    }
    split_pids = sorted(pid_to_split)
    if args.max_patients and args.max_patients > 0:
        split_pids = split_pids[: int(args.max_patients)]
        print(f"  --max-patients={args.max_patients}: keeping {len(split_pids):,} ids")

    print("Writing ACES-style labels…")
    write_task_labels(args.labels_dir, splits, args.labels_out)

    parquet_path = args.events_parquet or os.path.join(args.output_dir, "ehrshot.parquet")
    # Full runs convert CSV→parquet once. Tiny --max-patients debug scans the CSV
    # so we do not write a 3GB cache just to inspect a handful of patients.
    if args.max_patients and args.max_patients > 0 and not os.path.isfile(parquet_path):
        print(f"  debug scan of {args.events_csv} (no parquet cache)")
        events_lf = _scan_events(args.events_csv)
    else:
        print("Preparing events table…")
        _csv_to_parquet(args.events_csv, parquet_path)
        events_lf = _scan_events(parquet_path)

    print("Loading person-table demographics (Gender/Race/DOB)…")
    person = events_lf.filter(pl.col("omop_table") == "person").collect()
    demo_by_pid: dict[int, dict[str, Any]] = {}
    for pid, g in person.group_by("patient_id"):
        key = _group_pid(pid)
        rows = []
        for rec in g.iter_rows(named=True):
            rec = dict(rec)
            rec["start"] = _row_start(rec)
            rows.append(rec)
        demo_by_pid[key] = _person_demographics(rows)
    del person
    print(f"  demographics for {len(demo_by_pid):,} patients")

    raw_dir = os.path.join(args.output_dir, "_raw_strings")
    os.makedirs(raw_dir, exist_ok=True)
    print("Tokenising full trajectories in patient chunks (no event cap)…")
    counts: Counter[str] = Counter()
    raw_paths: list[str] = []
    n_empty = 0
    n_done = 0
    n_kept = 0
    n_tokens = 0
    chunk_idx = 0
    for sub in _iter_patient_chunks(events_lf, split_pids, chunk_size=args.chunk_size):
        recs: list[dict] = []
        for pid, g in sub.group_by("patient_id"):
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done:,}/{len(split_pids):,} patients…", flush=True)
            key = _group_pid(pid)
            split = pid_to_split.get(key)
            if split is None:
                continue
            rows = []
            for rec in g.sort("start").iter_rows(named=True):
                rec = dict(rec)
                rec["start"] = _row_start(rec)
                if rec["start"] is None:
                    continue
                rows.append(rec)
            tokens, times = tokenize_patient(rows, demo_by_pid.get(key, {}))
            if len(tokens) <= 1:
                n_empty += 1
                continue
            counts.update(tokens)
            recs.append(
                {
                    "subject_id": key,
                    "split": split,
                    "tokens": tokens,
                    "timestamps": times,
                }
            )
            n_kept += 1
            n_tokens += len(tokens)
        del sub
        if recs:
            path = os.path.join(raw_dir, f"chunk_{chunk_idx:04d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
            raw_paths.append(path)
            chunk_idx += 1

    print(
        f"  tokenised patients={n_kept:,} empty={n_empty} "
        f"mean_tokens={n_tokens / max(n_kept, 1):.0f}"
    )

    token_to_id = write_vocab(counts, os.path.join(args.output_dir, "vocab.csv"))
    print(f"  vocab size={len(token_to_id):,} → {args.output_dir}/vocab.csv")

    by_split: dict[str, list[dict]] = defaultdict(list)
    for path in raw_paths:
        with open(path, "rb") as f:
            recs = pickle.load(f)
        for rec in recs:
            by_split[rec["split"]].append(
                {
                    "subject_id": rec["subject_id"],
                    "tokens": [token_to_id.get(t, 0) for t in rec["tokens"]],
                    "timestamps": rec["timestamps"],
                }
            )
        if not args.keep_raw:
            os.remove(path)
    if not args.keep_raw:
        try:
            os.rmdir(raw_dir)
        except OSError:
            pass

    for split in ("train", "tuning", "held_out"):
        recs = by_split[split]
        recs.sort(key=lambda r: r["subject_id"])
        n_shards = _write_shards(recs, os.path.join(args.output_dir, split))
        print(f"  {split}: {len(recs):,} patients, {n_shards} shards")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="Tokenise EHRSHOT for tEHR")
    p.add_argument("--events-csv", default=os.path.join(DEFAULT_ASSETS, "data/ehrshot.csv"))
    p.add_argument(
        "--events-parquet",
        default="",
        help="Optional parquet cache of ehrshot.csv (default: sibling .parquet)",
    )
    p.add_argument(
        "--splits-csv",
        default=os.path.join(DEFAULT_ASSETS, "splits/person_id_map.csv"),
    )
    p.add_argument(
        "--labels-dir",
        default=os.path.join(DEFAULT_ASSETS, "benchmark"),
    )
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--labels-out", default=DEFAULT_LABELS)
    p.add_argument("--chunk-size", type=int, default=250)
    p.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="If >0, tokenise only the first N split patient ids (debug).",
    )
    p.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep intermediate string-token pickles under _raw_strings/.",
    )
    args = p.parse_args()
    args.events_parquet = args.events_parquet or None
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.labels_out, exist_ok=True)
    build(args)


if __name__ == "__main__":
    main()
