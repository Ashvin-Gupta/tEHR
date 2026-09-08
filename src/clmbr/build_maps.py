"""Download MIMIC-code / RxNorm tables and compile CLMBR lookup JSON.

python -m src.clmbr.build_maps
python -m src.clmbr.build_maps --report
python -m src.clmbr.build_maps --download-model

Optional Athena overlays (Maps-to, in-vocab SNOMED only):
  data/clmbr_maps/raw/icd9cm_to_snomed.tsv
  data/clmbr_maps/raw/icd10cm_to_snomed.tsv
Columns: source_code, snomed_code
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from typing import Optional

from src.clmbr.mapping import (
    MAR_STATUS,
    CodeMapper,
    NursingIndex,
    fallback_loinc,
    icd_lookup_keys,
    loinc_from_label,
    name_variants,
    normalize_name,
)
from src.clmbr.vocab import ClmbrVocab

PROJECT_DIR = os.environ.get("PROJECTDIR", "/projects/u6ps")

MIMIC_CODE = "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/mapping"
MAP_FILES = [
    "d_labitems_to_loinc.csv",
    "lab_itemid_to_loinc.csv",
    "inputevents_to_rxnorm.csv",
    "proc_itemid.csv",
    "outputevents_to_loinc.csv",
    "meas_chartevents_main.csv",
]
# Full RxNorm prescribe zip now redirects to UTS login. RxTerms is public.
RXTERMS_BASE = "https://data.lhncbc.nlm.nih.gov/public/rxterms/release"
HF_MODEL = "StanfordShahLab/clmbr-t-base"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
RXNAV = "https://rxnav.nlm.nih.gov/REST"
UA = "TextCancEHR2-clmbr-maps (research; StanfordShahLab/clmbr-t-base eval)"


def _request(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  exists {dest}")
        return
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_mimic_maps(maps_dir: str) -> None:
    raw = os.path.join(maps_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    for name in MAP_FILES:
        try:
            _download(f"{MIMIC_CODE}/{name}", os.path.join(raw, name))
        except Exception as exc:
            print(f"  WARN could not download {name}: {exc}")


def _valid_zip(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return bool(zf.namelist())
    except zipfile.BadZipFile:
        return False


def download_rxnorm(maps_dir: str) -> Optional[str]:
    """Download public RxTerms (names + ingredients → RXCUI). No UMLS license."""
    import datetime as _dt

    raw = os.path.join(maps_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    zpath = os.path.join(raw, "RxTerms_current.zip")
    if _valid_zip(zpath):
        print(f"  exists {zpath}")
        return zpath
    if os.path.isfile(zpath):
        os.remove(zpath)
    today = _dt.date.today().replace(day=1)
    for i in range(8):
        year = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year -= 1
        stamp = f"{year}{month:02d}"
        url = f"{RXTERMS_BASE}/RxTerms{stamp}.zip"
        try:
            _download(url, zpath)
        except Exception as exc:
            print(f"  skip {url}: {exc}")
            if os.path.isfile(zpath):
                os.remove(zpath)
            continue
        if _valid_zip(zpath):
            print(f"  RxTerms {stamp}")
            return zpath
        print(f"  not a zip: {url}")
        os.remove(zpath)
    print("  WARN could not download RxTerms")
    return None


def download_model(model_dir: str) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading {HF_MODEL} → {model_dir}")
    snapshot_download(HF_MODEL, local_dir=model_dir)


def _loinc_token(code: str) -> Optional[str]:
    code = (code or "").strip()
    if not code:
        return None
    return f"LOINC/{code}"


def _omop_rx_token(vocab_id: str, code: str) -> Optional[str]:
    vocab_id = (vocab_id or "").strip()
    code = (code or "").strip()
    if not code:
        return None
    if vocab_id == "RxNorm":
        return f"RxNorm/{code}"
    if vocab_id == "RxNorm Extension":
        return f"RxNorm Extension/{code}"
    return None


def _itemid_from_row(row: dict) -> Optional[str]:
    for cand in ("itemid", "itemid (omop_source_code)", "omop_source_code"):
        if cand in row and row[cand]:
            try:
                return str(int(row[cand]))
            except ValueError:
                return str(row[cand]).strip()
    return None


def _expand_icd_keys(src: str, dest: dict, tok: str) -> None:
    for key in icd_lookup_keys(src):
        dest.setdefault(key, tok)


def download_wikidata_icd_maps(raw: str, vocab: ClmbrVocab) -> tuple[dict, dict]:
    """License-free ICD ↔ SNOMED links from Wikidata (incomplete vs Athena)."""
    icd9, icd10 = {}, {}
    queries = (
        (
            icd10,
            """SELECT ?icd ?snomed WHERE {
              ?item wdt:P1690 ?icd .
              ?item wdt:P5806 ?snomed .
            }""",
        ),
        (
            icd9,
            """SELECT ?icd ?snomed WHERE {
              ?item wdt:P1692 ?icd .
              ?item wdt:P5806 ?snomed .
            }""",
        ),
    )
    for dest, sparql in queries:
        url = WIKIDATA_SPARQL + "?" + urllib.parse.urlencode(
            {"format": "json", "query": sparql}
        )
        try:
            payload = json.loads(_request(url, timeout=180).decode("utf-8"))
        except Exception as exc:
            print(f"  WARN Wikidata SPARQL failed ({exc}); retrying in 15s")
            time.sleep(15)
            try:
                payload = json.loads(_request(url, timeout=180).decode("utf-8"))
            except Exception as exc2:
                print(f"  WARN Wikidata SPARQL failed again: {exc2}")
                continue
        n = 0
        for row in payload.get("results", {}).get("bindings", []):
            icd = (row.get("icd") or {}).get("value") or ""
            snomed = (row.get("snomed") or {}).get("value") or ""
            icd = icd.rsplit("/", 1)[-1].strip()
            snomed = snomed.rsplit("/", 1)[-1].strip()
            if not icd or not snomed:
                continue
            tok = snomed if snomed.startswith("SNOMED/") else f"SNOMED/{snomed}"
            if vocab.contains(tok):
                _expand_icd_keys(icd, dest, tok)
                n += 1
        print(f"  Wikidata in-vocab maps: {n} rows → {len(dest)} keys")
    # Persist so later compiles skip SPARQL if desired
    for name, dest in (("icd9cm_wikidata.tsv", icd9), ("icd10cm_wikidata.tsv", icd10)):
        path = os.path.join(raw, name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source_code", "snomed_code"])
            for src, tok in sorted(dest.items()):
                w.writerow([src, tok])
    return icd9, icd10


def _add_rx_name(dest: dict[str, str], name: str, tok: Optional[str], prefer: bool = False) -> None:
    if not tok or not name:
        return
    for key in name_variants(name):
        if prefer:
            dest[key] = tok
        else:
            dest.setdefault(key, tok)


def _parse_rxterms(zpath: str, vocab: ClmbrVocab) -> dict[str, str]:
    """Public RxTerms → in-vocab RxNorm tokens.

    CLMBR-T stores many clinical-drug RXCUIs (SCD/SBD) but often not the
    ingredient CUI, so ingredient names are rolled up to an in-vocab SCD.
    """
    rxcui_by_name: dict[str, str] = {}
    ing_to_scd: dict[str, list[str]] = {}
    print("  parsing RxTerms names / ingredients")
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        ing = next((n for n in names if "ingredient" in n.lower() and n.lower().endswith(".txt")), None)
        main = next(
            (
                n
                for n in names
                if n.lower().endswith(".txt")
                and "rxterms" in n.lower()
                and "archive" not in n.lower()
                and "ingredient" not in n.lower()
                and "release" not in n.lower()
            ),
            None,
        )
        if ing:
            with zf.open(ing) as rawf:
                text = io.TextIOWrapper(rawf, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text, delimiter="|")
                for row in reader:
                    name = (row.get("INGREDIENT") or "").strip()
                    if not name:
                        continue
                    for cui in ((row.get("ING_RXCUI") or "").strip(), (row.get("RXCUI") or "").strip()):
                        if not cui:
                            continue
                        tok = f"RxNorm/{cui}"
                        if vocab.contains(tok):
                            ing_to_scd.setdefault(name.lower(), []).append(tok)
        for name, toks in ing_to_scd.items():
            _add_rx_name(rxcui_by_name, name, toks[0], prefer=True)
        if main:
            with zf.open(main) as rawf:
                text = io.TextIOWrapper(rawf, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text, delimiter="|")
                for row in reader:
                    tok = None
                    for cui_key in ("RXCUI", "GENERIC_RXCUI"):
                        cui = (row.get(cui_key) or "").strip()
                        cand = f"RxNorm/{cui}" if cui else None
                        if cand and vocab.contains(cand):
                            tok = cand
                            break
                    if not tok:
                        continue
                    for field in (
                        "BRAND_NAME",
                        "DISPLAY_NAME",
                        "DISPLAY_NAME_SYNONYM",
                        "FULL_GENERIC_NAME",
                        "FULL_NAME",
                        "PSN",
                    ):
                        _add_rx_name(rxcui_by_name, row.get(field) or "", tok)
                    generic = row.get("FULL_GENERIC_NAME") or ""
                    first = normalize_name(generic).split(" ")[0] if generic else ""
                    if first:
                        rxcui_by_name.setdefault(first, tok)
    print(f"  RxTerms names={len(rxcui_by_name)}")
    return rxcui_by_name


def _unique_med_names(meds_dir: str) -> list[str]:
    import polars as pl
    from pathlib import Path

    names: set[str] = set()
    for split in ("held_out", "tuning", "train"):
        shard_dir = Path(meds_dir) / split
        if not shard_dir.is_dir():
            continue
        files = sorted(shard_dir.glob("*.parquet"))
        print(f"  scanning unique medication strings in {split} ({len(files)} shards)")
        for i, path in enumerate(files, 1):
            df = pl.read_parquet(path, columns=["code"])
            med = df.filter(pl.col("code").str.starts_with("MEDICATION//"))
            for code in med["code"].unique().to_list():
                if not code:
                    continue
                parts = [p for p in code.split("//") if p]
                if len(parts) >= 3 and parts[1] in {"START", "STOP"}:
                    names.add(parts[2])
                elif len(parts) >= 2:
                    names.add(parts[1])
            if i % 50 == 0 or i == len(files):
                print(f"    {i}/{len(files)} unique_names={len(names)}")
        # held_out + tuning is enough for coverage; train is large.
        if split == "tuning":
            break
    return sorted(names)


def _rxnav_resolve(name: str, vocab: ClmbrVocab) -> Optional[str]:
    variants = [name] + name_variants(name)
    seen_cuis: list[str] = []
    for term in variants[:4]:
        if not term:
            continue
        q = urllib.parse.quote(term)
        try:
            exact = json.loads(
                _request(f"{RXNAV}/rxcui.json?name={q}&search=2", timeout=30).decode()
            )
            ids = (exact.get("idGroup") or {}).get("rxnormId") or []
            if isinstance(ids, str):
                ids = [ids]
            for rxcui in ids:
                if rxcui:
                    seen_cuis.append(str(rxcui))
            approx = json.loads(
                _request(
                    f"{RXNAV}/approximateTerm.json?term={q}&maxEntries=6", timeout=30
                ).decode()
            )
            cands = (approx.get("approximateGroup") or {}).get("candidate") or []
            if isinstance(cands, dict):
                cands = [cands]
            for cand in cands:
                cui = str(cand.get("rxcui") or "")
                if cui:
                    seen_cuis.append(cui)
        except Exception:
            continue
        time.sleep(0.05)
    for cui in seen_cuis:
        tok = f"RxNorm/{cui}"
        if vocab.contains(tok):
            return tok
        try:
            related = json.loads(
                _request(f"{RXNAV}/rxcui/{cui}/allrelated.json", timeout=30).decode()
            )
        except Exception:
            continue
        time.sleep(0.05)
        groups = ((related.get("allRelatedGroup") or {}).get("conceptGroup") or [])
        tty_rank = {"SCD": 0, "SBD": 1, "GPCK": 2, "BPCK": 3, "IN": 4, "PIN": 5, "BN": 6}
        ranked = []
        for group in groups:
            tty = group.get("tty") or ""
            for props in group.get("conceptProperties") or []:
                tok = f"RxNorm/{props.get('rxcui')}"
                if vocab.contains(tok):
                    ranked.append((tty_rank.get(tty, 9), tok))
        if ranked:
            ranked.sort()
            return ranked[0][1]
    return None


def resolve_unmatched_meds(
    maps_dir: str,
    rxcui_by_name: dict[str, str],
    meds_dir: str,
    vocab: ClmbrVocab,
    max_calls: int = 800,
) -> None:
    cache_path = os.path.join(maps_dir, "raw", "rxnav_cache.json")
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    names = _unique_med_names(meds_dir)
    pending = []
    for name in names:
        if normalize_name(name) in MAR_STATUS:
            continue
        keys = name_variants(name)
        if any(k in rxcui_by_name for k in keys):
            continue
        pending.append(name)
    print(f"  RxNav: {len(pending)} unmatched unique names (cap {max_calls})")
    hits = 0
    for i, name in enumerate(pending[:max_calls], 1):
        if name in cache:
            tok = cache[name]
        else:
            tok = _rxnav_resolve(name, vocab)
            cache[name] = tok
            if i % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                print(f"    RxNav {i}/{min(len(pending), max_calls)} hits={hits}")
        if tok:
            hits += 1
            for key in name_variants(name):
                rxcui_by_name.setdefault(key, tok)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"  RxNav resolved {hits} names")


def compile_lookups(
    maps_dir: str,
    vocab: ClmbrVocab,
    meds_dir: Optional[str] = None,
    resolve_rxnav: bool = True,
) -> dict:
    raw = os.path.join(maps_dir, "raw")
    lab_itemid: dict[str, str] = {}
    lab_pair: dict[str, str] = {}
    lab_label: dict[str, str] = {}
    infusion: dict[str, str] = {}
    proc_itemid: dict[str, str] = {}
    output_itemid: dict[str, str] = {}
    rxcui_by_name: dict[str, str] = {}

    def add_lab(itemid: str, tok: Optional[str], unit: str = "", label: str = "") -> None:
        if tok and not vocab.contains(tok) and tok.startswith("LOINC/"):
            tok = fallback_loinc(tok.split("/", 1)[1])
        if not tok or not vocab.contains(tok):
            tok = loinc_from_label(label or "")
        tok = tok if tok and vocab.contains(tok) else None
        if not tok:
            return
        lab_itemid.setdefault(itemid, tok)
        if unit:
            lab_pair.setdefault(f"{itemid}||{unit}", tok)
        key = normalize_name(label)
        if key:
            lab_label.setdefault(key, tok)

    omop_lab = os.path.join(raw, "d_labitems_to_loinc.csv")
    if os.path.isfile(omop_lab):
        with open(omop_lab, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("omop_vocabulary_id") or "").strip() != "LOINC":
                    continue
                itemid = _itemid_from_row(row)
                if not itemid:
                    continue
                add_lab(
                    itemid,
                    _loinc_token(row.get("omop_concept_code") or ""),
                    (row.get("valueuom") or "").strip(),
                    row.get("label") or "",
                )

    simple_lab = os.path.join(raw, "lab_itemid_to_loinc.csv")
    if os.path.isfile(simple_lab):
        with open(simple_lab, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    itemid = str(int(row["itemid"]))
                except (KeyError, ValueError):
                    continue
                add_lab(
                    itemid,
                    _loinc_token(row.get("loinc") or ""),
                    (row.get("valueuom") or "").strip(),
                    row.get("label") or "",
                )

    zpath = os.path.join(raw, "RxTerms_current.zip")
    if _valid_zip(zpath):
        parsed = _parse_rxterms(zpath, vocab)
        for k, v in parsed.items():
            rxcui_by_name.setdefault(k, v)
    elif os.path.isfile(zpath):
        print(f"  WARN ignoring invalid RxTerms zip at {zpath}")

    inf_path = os.path.join(raw, "inputevents_to_rxnorm.csv")
    if os.path.isfile(inf_path):
        with open(inf_path, newline="") as f:
            for row in csv.DictReader(f):
                tok = _omop_rx_token(
                    row.get("omop_vocabulary_id") or "",
                    row.get("omop_concept_code") or "",
                )
                itemid = _itemid_from_row(row)
                if tok and itemid and vocab.contains(tok):
                    infusion.setdefault(itemid, tok)
                elif itemid:
                    for name in (row.get("label"), row.get("omop_concept_name")):
                        for key in name_variants(name or ""):
                            alt = rxcui_by_name.get(key)
                            if alt and vocab.contains(alt):
                                infusion.setdefault(itemid, alt)
                                break
                for name in (row.get("label"), row.get("omop_concept_name")):
                    for key in name_variants(name or ""):
                        if tok and vocab.contains(tok):
                            rxcui_by_name.setdefault(key, tok)

    for fname, dest in (
        ("proc_itemid.csv", proc_itemid),
        ("outputevents_to_loinc.csv", output_itemid),
        ("meas_chartevents_main.csv", lab_itemid),
    ):
        path = os.path.join(raw, fname)
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                itemid = _itemid_from_row(row)
                vocab_id = (row.get("omop_vocabulary_id") or "").strip()
                code = (row.get("omop_concept_code") or "").strip()
                if not itemid or not vocab_id or not code:
                    continue
                tok = f"{vocab_id}/{code}"
                if vocab_id == "LOINC" and not vocab.contains(tok):
                    tok = fallback_loinc(code) or tok
                if not vocab.contains(tok):
                    alt = loinc_from_label(row.get("label") or "")
                    if alt:
                        tok = alt
                if not vocab.contains(tok):
                    continue
                dest.setdefault(itemid, tok)
                if vocab_id == "LOINC" or tok.startswith("LOINC/"):
                    add_lab(itemid, tok, "", row.get("label") or "")

    labitems = os.path.join(raw, "d_labitems.csv.gz")
    if os.path.isfile(labitems):
        with gzip.open(labitems, "rt") as f:
            for row in csv.DictReader(f):
                try:
                    itemid = str(int(row["itemid"]))
                except (KeyError, ValueError):
                    continue
                if itemid in lab_itemid:
                    continue
                key = normalize_name(row.get("label") or "")
                if key and key in lab_label:
                    lab_itemid[itemid] = lab_label[key]

    ditems = os.path.join(raw, "d_items.csv.gz")
    if os.path.isfile(ditems):
        with gzip.open(ditems, "rt") as f:
            for row in csv.DictReader(f):
                try:
                    itemid = str(int(row["itemid"]))
                except (KeyError, ValueError):
                    continue
                label = row.get("label") or ""
                key = normalize_name(label)
                if itemid not in lab_itemid:
                    tok = loinc_from_label(label)
                    if tok and vocab.contains(tok):
                        lab_itemid[itemid] = tok
                    elif key and key in lab_label:
                        lab_itemid[itemid] = lab_label[key]
                if itemid not in infusion:
                    for k in name_variants(label):
                        tok = rxcui_by_name.get(k)
                        if tok:
                            infusion[itemid] = tok
                            break

    print("Downloading Wikidata ICD→SNOMED (in-vocab only)…")
    icd9cm, icd10cm = download_wikidata_icd_maps(raw, vocab)
    extra9, extra10 = _optional_icd_maps(raw, vocab)
    icd9cm.update(extra9)
    icd10cm.update(extra10)

    if resolve_rxnav and meds_dir:
        print("Resolving leftover medication names via NLM RxNav…")
        resolve_unmatched_meds(maps_dir, rxcui_by_name, meds_dir, vocab)

    lookups = {
        "lab_itemid": lab_itemid,
        "lab_itemid_unit": lab_pair,
        "lab_label": lab_label,
        "infusion_itemid": infusion,
        "proc_itemid": proc_itemid,
        "output_itemid": output_itemid,
        "rxcui_by_name": rxcui_by_name,
        "icd9cm": icd9cm,
        "icd10cm": icd10cm,
    }
    out = os.path.join(maps_dir, "lookups.json.gz")
    with gzip.open(out, "wt") as f:
        json.dump(lookups, f)
    print(
        f"Wrote {out}: labs={len(lab_itemid)} infusions={len(infusion)} "
        f"rx_names={len(rxcui_by_name)} proc={len(proc_itemid)} "
        f"output={len(output_itemid)} icd9={len(icd9cm)} icd10={len(icd10cm)}"
    )
    return lookups


def _optional_icd_maps(raw: str, vocab: ClmbrVocab) -> tuple[dict, dict]:
    """Load optional TSV dumps: icd9cm_to_snomed.tsv / icd10cm_to_snomed.tsv."""
    icd9, icd10 = {}, {}
    for name, dest in (
        ("icd9cm_to_snomed.tsv", icd9),
        ("icd10cm_to_snomed.tsv", icd10),
    ):
        path = os.path.join(raw, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            src_col = reader.fieldnames[0] if reader.fieldnames else "source_code"
            tgt_col = (
                reader.fieldnames[1]
                if reader.fieldnames and len(reader.fieldnames) > 1
                else "snomed_code"
            )
            for row in reader:
                src = (row.get(src_col) or "").strip()
                tgt = (row.get(tgt_col) or "").strip()
                if not src or not tgt:
                    continue
                tok = tgt if tgt.startswith("SNOMED/") else f"SNOMED/{tgt}"
                if vocab.contains(tok):
                    _expand_icd_keys(src, dest, tok)
        print(f"  loaded {name}: {len(dest)} in-vocab Maps-to keys")
    return icd9, icd10


def _pct(a: int, b: int) -> float:
    return 0.0 if b == 0 else 100.0 * a / b


def vocab_json_coverage(
    vocab_json_path: str, mapper: CodeMapper, nursing: Optional[NursingIndex] = None
) -> dict:
    with open(vocab_json_path) as f:
        payload = json.load(f)
    codes = [c for c in (payload.get("code_to_idx") or {}) if c and c != "<UNK>"]
    stage = Counter()
    cat_n, cat_hit = Counter(), Counter()
    nursing_n = nursing_hit = 0
    for code in codes:
        result = mapper.map_code(code)
        stage[result.stage] += 1
        cat_n[result.category] += 1
        if result.stage == "in_vocab":
            cat_hit[result.category] += 1
        if nursing is not None and nursing.is_nursing(code):
            nursing_n += 1
            if result.stage == "in_vocab":
                nursing_hit += 1
    n = len(codes)
    clinical_n = n - nursing_n
    clinical_hit = stage["in_vocab"] - nursing_hit
    return {
        "path": vocab_json_path,
        "codes": n,
        "in_vocab": stage["in_vocab"],
        "in_vocab_pct": round(_pct(stage["in_vocab"], n), 2),
        "oov": stage["oov"],
        "oov_pct": round(_pct(stage["oov"], n), 2),
        "no_map": stage["no_map"],
        "no_map_pct": round(_pct(stage["no_map"], n), 2),
        "nursing_codes": nursing_n,
        "clinical_codes": clinical_n,
        "clinical_in_vocab_pct": round(_pct(clinical_hit, clinical_n), 2),
        "by_category": [
            {
                "category": c,
                "codes": cat_n[c],
                "in_vocab_pct": round(_pct(cat_hit[c], cat_n[c]), 2),
            }
            for c, _ in cat_n.most_common()
        ],
    }


def coverage_report(
    meds_dir: str,
    maps_dir: str,
    model_dir: str,
    split: str = "held_out",
    vocab_json: Optional[str] = None,
) -> dict:
    import polars as pl
    from pathlib import Path

    vocab = ClmbrVocab.from_pretrained(model_dir)
    mapper = CodeMapper.load(maps_dir, vocab)
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parent = os.path.dirname(repo)
    nursing = NursingIndex.load(
        os.path.join(maps_dir, "raw", "d_items.csv.gz"),
        os.path.join(PROJECT_DIR, "MIMIC_data", "hosp", "d_items.csv.gz"),
        os.path.join(maps_dir, "raw", "d_labitems.csv.gz"),
        os.path.join(PROJECT_DIR, "MIMIC_data", "hosp", "d_labitems.csv.gz"),
    )
    vocab_json = vocab_json or os.path.join(parent, "vocab.json")

    cat_n, cat_hit = Counter(), Counter()
    stage = Counter()
    unique_stage = Counter()
    unmapped = Counter()
    n = 0
    n_nursing = n_nursing_hit = 0
    unique_codes = 0
    shard_dir = Path(meds_dir) / split
    files = sorted(shard_dir.glob("*.parquet"))
    print(f"Coverage on {split}: {len(files)} shards")
    seen_codes: set[str] = set()
    for i, path in enumerate(files, 1):
        df = pl.read_parquet(path, columns=["code"])
        vc = df.group_by("code").len()
        for rec in vc.iter_rows(named=True):
            code = rec.get("code")
            cnt = int(rec["len"])
            if code is None:
                continue
            n += cnt
            result = mapper.map_code(code)
            cat_n[result.category] += cnt
            stage[result.stage] += cnt
            if result.stage == "in_vocab":
                cat_hit[result.category] += cnt
            elif result.stage == "no_map":
                unmapped[f"{result.category}\t{code}"] += cnt
            if nursing.is_nursing(code):
                n_nursing += cnt
                if result.stage == "in_vocab":
                    n_nursing_hit += cnt
            if code not in seen_codes:
                seen_codes.add(code)
                unique_codes += 1
                unique_stage[result.stage] += 1
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    n_clinical = n - n_nursing
    clinical_hit = stage["in_vocab"] - n_nursing_hit
    vj = vocab_json_coverage(vocab_json, mapper, nursing) if os.path.isfile(vocab_json) else None

    report = {
        "split": split,
        "vocab_json": vj,
        "events": {
            "n": n,
            "in_vocab": stage["in_vocab"],
            "in_vocab_pct": round(_pct(stage["in_vocab"], n), 2),
            "oov_pct": round(_pct(stage["oov"], n), 2),
            "no_map_pct": round(_pct(stage["no_map"], n), 2),
            "nursing_n": n_nursing,
            "nursing_share_pct": round(_pct(n_nursing, n), 2),
            "excluding_nursing": {
                "n": n_clinical,
                "in_vocab": clinical_hit,
                "in_vocab_pct": round(_pct(clinical_hit, n_clinical), 2),
            },
        },
        "unique_meds_codes": {
            "n": unique_codes,
            "in_vocab": unique_stage["in_vocab"],
            "in_vocab_pct": round(_pct(unique_stage["in_vocab"], unique_codes), 2),
        },
        "by_category": [
            {
                "category": c,
                "events": cat_n[c],
                "in_vocab_pct": round(_pct(cat_hit[c], cat_n[c]), 2),
            }
            for c, _ in cat_n.most_common()
        ],
    }
    # Back-compat aliases used by older notes / scripts.
    report["in_vocab_pct"] = report["events"]["in_vocab_pct"]
    report["oov_pct"] = report["events"]["oov_pct"]
    report["no_map_pct"] = report["events"]["no_map_pct"]

    out = os.path.join(maps_dir, f"coverage_{split}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    top_path = os.path.join(maps_dir, f"unmapped_top_{split}.json")
    with open(top_path, "w") as f:
        json.dump(
            [
                {"category": k.split("\t", 1)[0], "code": k.split("\t", 1)[1], "n": v}
                for k, v in unmapped.most_common(80)
            ],
            f,
            indent=2,
        )
    print_coverage_summary(report)
    print("Wrote", out, "and", top_path)
    return report


def print_coverage_summary(report: dict) -> None:
    vj = report.get("vocab_json") or {}
    ev = report.get("events") or {}
    ex = ev.get("excluding_nursing") or {}
    print("=== Mapping coverage ===")
    if vj:
        print(
            f"vocab.json codes mapped:  {vj.get('in_vocab', 0):,}/{vj.get('codes', 0):,} "
            f"({vj.get('in_vocab_pct')}%)"
        )
    print(
        f"events mapped:            {ev.get('in_vocab', 0):,}/{ev.get('n', 0):,} "
        f"({ev.get('in_vocab_pct')}%)"
    )
    print(
        f"events mapped, no nursing:{ex.get('in_vocab', 0):,}/{ex.get('n', 0):,} "
        f"({ex.get('in_vocab_pct')}%)"
        f"  [nursing share {ev.get('nursing_share_pct')}% of events]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--meds-dir", default=None)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print coverage (vocab.json + events ± nursing) without recompiling maps",
    )
    parser.add_argument("--vocab-json", default=None, help="Frequency vocab.json for code-level coverage")
    parser.add_argument("--split", default="held_out")
    parser.add_argument("--labitems", default=None, help="MIMIC d_labitems.csv.gz for label fallback")
    parser.add_argument("--skip-rxnav", action="store_true")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    maps_dir = args.maps_dir or os.path.join(repo, "data", "clmbr_maps")
    model_dir = args.model_dir or os.path.join(repo, "data", "clmbr_t_base")
    meds_dir = args.meds_dir or os.path.join(PROJECT_DIR, "clean_meds")

    os.makedirs(maps_dir, exist_ok=True)
    if args.download_model:
        download_model(model_dir)

    if not args.report_only:
        print("Downloading mapping tables…")
        download_mimic_maps(maps_dir)
        download_rxnorm(maps_dir)
        labitems = args.labitems or os.path.join(
            PROJECT_DIR, "MIMIC_data", "hosp", "d_labitems.csv.gz"
        )
        if os.path.isfile(labitems):
            dest = os.path.join(maps_dir, "raw", "d_labitems.csv.gz")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.isfile(dest):
                shutil.copy2(labitems, dest)
                print(f"  copied {labitems}")
        ditems = os.path.join(PROJECT_DIR, "MIMIC_data", "hosp", "d_items.csv.gz")
        if os.path.isfile(ditems):
            dest = os.path.join(maps_dir, "raw", "d_items.csv.gz")
            if not os.path.isfile(dest):
                shutil.copy2(ditems, dest)
                print(f"  copied {ditems}")

        if not os.path.isdir(model_dir) or not os.path.isfile(
            os.path.join(model_dir, "dictionary.msgpack")
        ):
            print("Model dictionary missing; downloading CLMBR-T-base")
            download_model(model_dir)

        vocab = ClmbrVocab.from_pretrained(model_dir)
        compile_lookups(
            maps_dir,
            vocab,
            meds_dir=meds_dir,
            resolve_rxnav=not args.skip_rxnav,
        )

    if args.report or args.report_only:
        coverage_report(
            meds_dir,
            maps_dir,
            model_dir,
            split=args.split,
            vocab_json=args.vocab_json,
        )


if __name__ == "__main__":
    main()
