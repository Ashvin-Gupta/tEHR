"""Map clean_meds MIMIC codes onto CLMBR-T OMOP strings."""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.clmbr.vocab import ClmbrVocab

BP_LOINC = {
    "Blood Pressure Systolic": "LOINC/8480-6",
    "Blood Pressure Diastolic": "LOINC/8462-4",
}
NAMED_LOINC = {
    "BMI": "LOINC/39156-5",
}
WEIGHT_LOINC = "LOINC/29463-7"

# CDC Recode: 1=AIAN, 2=Asian, 3=Black, 4=NHPI, 5=White
RACE_TO_CLMBR = {
    "WHITE": "Race/5",
    "BLACK": "Race/3",
    "ASIAN": "Race/2",
    "HISPANIC": "Ethnicity/Hispanic",
}
VISIT_RULES = (
    ("HOSPITAL_ADMISSION", "Visit/IP"),
    ("ICU_ADMISSION", "Visit/IP"),
    ("ED_REGISTRATION", "Visit/ER"),
    ("HOSPITAL_DISCHARGE", "Visit/IP"),
    ("ICU_DISCHARGE", "Visit/IP"),
    ("ED_OUT", "Visit/ER"),
)

# mimic-code LOINCs that CLMBR-T does not store; parent codes that it does.
LOINC_FALLBACK = {
    "59408-5": "20564-1",  # SpO2 pulse ox → O2 sat in blood
    "2708-6": "20564-1",  # arterial O2 sat
    "76534-7": "8480-6",  # NIBP systolic → systolic BP
    "76535-4": "8462-4",
    "76536-2": "8478-0",
    "1963-8": "2028-9",  # bicarbonate → CO2
    "76528-9": "9279-1",  # RR spontaneous
    "19834-1": "9279-1",  # RR set
}

LABEL_LOINC = (
    (re.compile(r"o2 saturation|spo2|pulse\s*ox", re.I), "LOINC/20564-1"),
    (re.compile(r"non invasive blood pressure systolic", re.I), "LOINC/8480-6"),
    (re.compile(r"non invasive blood pressure diastolic", re.I), "LOINC/8462-4"),
    (re.compile(r"non invasive blood pressure mean", re.I), "LOINC/8478-0"),
    (re.compile(r"^bicarbonate$|hco3", re.I), "LOINC/2028-9"),
    (re.compile(r"respiratory rate", re.I), "LOINC/9279-1"),
    (re.compile(r"heart rate", re.I), "LOINC/8867-4"),
    (re.compile(r"temperature fahrenheit|temperature celsius", re.I), "LOINC/8310-5"),
    (re.compile(r"inspired o2|fio2", re.I), "LOINC/3150-0"),
    (re.compile(r"glucose finger|fingerstick glucose|poc glucose", re.I), "LOINC/2339-0"),
)

# High-frequency MIMIC ICD → in-vocab SNOMED, including ICD *parents*
# (3-digit ICD-9 / 3-char ICD-10). OHDSI/MIMIC does the same walk via Athena
# CONCEPT + CONCEPT_RELATIONSHIP 'Maps to'; we do not have that dump, so
# parents of the source ICD code are looked up here and in compiled TSVs.
ICD9_SNOMED = {
    "038": "SNOMED/91302008",
    "244": "SNOMED/40930008",
    "2449": "SNOMED/40930008",
    "250": "SNOMED/44054006",
    "25000": "SNOMED/44054006",
    "272": "SNOMED/55822004",
    "2724": "SNOMED/55822004",
    "276": "SNOMED/237841007",
    "2761": "SNOMED/389087006",
    "278": "SNOMED/49601007",
    "27800": "SNOMED/49601007",
    "285": "SNOMED/271737000",
    "2859": "SNOMED/271737000",
    "300": "SNOMED/197480006",
    "3051": "SNOMED/89765005",
    "311": "SNOMED/35489007",
    "32723": "SNOMED/78275009",
    "401": "SNOMED/38341003",
    "4011": "SNOMED/38341003",
    "4019": "SNOMED/38341003",
    "403": "SNOMED/431855005",
    "40390": "SNOMED/431855005",
    "410": "SNOMED/22298006",
    "41071": "SNOMED/22298006",
    "411": "SNOMED/53741008",
    "412": "SNOMED/22298006",
    "413": "SNOMED/194828000",
    "414": "SNOMED/53741008",
    "41401": "SNOMED/53741008",
    "415": "SNOMED/59282003",
    "424": "SNOMED/368009",
    "425": "SNOMED/85898001",
    "427": "SNOMED/698247007",
    "42731": "SNOMED/49436004",
    "428": "SNOMED/42343007",
    "4280": "SNOMED/42343007",
    "434": "SNOMED/230690007",
    "43491": "SNOMED/230690007",
    "436": "SNOMED/230690007",
    "440": "SNOMED/399957001",
    "486": "SNOMED/233604007",
    "491": "SNOMED/13645005",
    "492": "SNOMED/13645005",
    "493": "SNOMED/195967001",
    "49390": "SNOMED/195967001",
    "496": "SNOMED/13645005",
    "507": "SNOMED/196151003",
    "511": "SNOMED/60046008",
    "514": "SNOMED/19242006",
    "518": "SNOMED/65710008",
    "51881": "SNOMED/65710008",
    "530": "SNOMED/235595009",
    "53081": "SNOMED/266433003",
    "560": "SNOMED/281647001",
    "562": "SNOMED/197210001",
    "571": "SNOMED/19943007",
    "577": "SNOMED/197456007",
    "578": "SNOMED/74474003",
    "584": "SNOMED/14669001",
    "5849": "SNOMED/14669001",
    "585": "SNOMED/709044004",
    "5856": "SNOMED/46177005",
    "586": "SNOMED/42369001",
    "599": "SNOMED/68566005",
    "5990": "SNOMED/68566005",
    "715": "SNOMED/396275006",
    "780": "SNOMED/271782001",
    "99591": "SNOMED/91302008",
    "99592": "SNOMED/76571007",
    "V1582": "SNOMED/266569009",
    "V45": "SNOMED/428987008",
    "V58": "SNOMED/182836005",
}
ICD10_SNOMED = {
    "A41": "SNOMED/91302008",
    "A419": "SNOMED/91302008",
    "C34": "SNOMED/93880001",
    "D62": "SNOMED/413532003",
    "D64": "SNOMED/271737000",
    "D649": "SNOMED/271737000",
    "E03": "SNOMED/40930008",
    "E039": "SNOMED/40930008",
    "E10": "SNOMED/46635009",
    "E11": "SNOMED/44054006",
    "E1122": "SNOMED/44054006",
    "E1165": "SNOMED/44054006",
    "E119": "SNOMED/44054006",
    "E66": "SNOMED/49601007",
    "E6601": "SNOMED/49601007",
    "E669": "SNOMED/49601007",
    "E78": "SNOMED/55822004",
    "E785": "SNOMED/55822004",
    "E86": "SNOMED/34095006",
    "E87": "SNOMED/237841007",
    "E871": "SNOMED/389087006",
    "E872": "SNOMED/3092008",
    "E875": "SNOMED/14140009",
    "E8770": "SNOMED/237841007",
    "F32": "SNOMED/35489007",
    "F329": "SNOMED/35489007",
    "F33": "SNOMED/35489007",
    "F41": "SNOMED/197480006",
    "F419": "SNOMED/197480006",
    "G47": "SNOMED/78275009",
    "G4733": "SNOMED/78275009",
    "G93": "SNOMED/2776000",
    "I10": "SNOMED/38341003",
    "I11": "SNOMED/38341003",
    "I12": "SNOMED/431855005",
    "I13": "SNOMED/431855005",
    "I20": "SNOMED/194828000",
    "I21": "SNOMED/22298006",
    "I214": "SNOMED/22298006",
    "I22": "SNOMED/22298006",
    "I25": "SNOMED/53741008",
    "I251": "SNOMED/53741008",
    "I2510": "SNOMED/53741008",
    "I252": "SNOMED/53741008",
    "I26": "SNOMED/59282003",
    "I27": "SNOMED/70995007",
    "I34": "SNOMED/48724000",
    "I35": "SNOMED/60573004",
    "I42": "SNOMED/85898001",
    "I48": "SNOMED/49436004",
    "I480": "SNOMED/49436004",
    "I4891": "SNOMED/49436004",
    "I49": "SNOMED/698247007",
    "I50": "SNOMED/42343007",
    "I5022": "SNOMED/42343007",
    "I5032": "SNOMED/42343007",
    "I509": "SNOMED/42343007",
    "I63": "SNOMED/230690007",
    "I639": "SNOMED/230690007",
    "I64": "SNOMED/230690007",
    "I70": "SNOMED/399957001",
    "I95": "SNOMED/45007003",
    "J15": "SNOMED/233604007",
    "J18": "SNOMED/233604007",
    "J189": "SNOMED/233604007",
    "J44": "SNOMED/13645005",
    "J441": "SNOMED/13645005",
    "J449": "SNOMED/13645005",
    "J45": "SNOMED/195967001",
    "J45909": "SNOMED/195967001",
    "J69": "SNOMED/196151003",
    "J80": "SNOMED/67782005",
    "J90": "SNOMED/60046008",
    "J96": "SNOMED/65710008",
    "J9600": "SNOMED/65710008",
    "J9601": "SNOMED/65710008",
    "K21": "SNOMED/235595009",
    "K219": "SNOMED/235595009",
    "K25": "SNOMED/397825006",
    "K52": "SNOMED/62315008",
    "K56": "SNOMED/281647001",
    "K57": "SNOMED/197210001",
    "K59": "SNOMED/14760008",
    "K72": "SNOMED/59927004",
    "K74": "SNOMED/19943007",
    "K85": "SNOMED/197456007",
    "K92": "SNOMED/74474003",
    "N17": "SNOMED/14669001",
    "N179": "SNOMED/14669001",
    "N18": "SNOMED/709044004",
    "N1830": "SNOMED/709044004",
    "N186": "SNOMED/46177005",
    "N189": "SNOMED/709044004",
    "N39": "SNOMED/68566005",
    "N390": "SNOMED/68566005",
    "N40": "SNOMED/90560007",
    "R05": "SNOMED/49727002",
    "R06": "SNOMED/267036007",
    "R13": "SNOMED/40739000",
    "R55": "SNOMED/271594007",
    "R57": "SNOMED/27942005",
    "R65": "SNOMED/91302008",
    "R6521": "SNOMED/76571007",
    "Z20": "SNOMED/840539006",
    "Z20822": "SNOMED/840539006",
    "Z51": "SNOMED/182836005",
    "Z66": "SNOMED/304253006",
    "Z79": "SNOMED/182836005",
    "Z7901": "SNOMED/698065002",
    "Z7902": "SNOMED/182836005",
    "Z86": "SNOMED/417662000",
    "Z8673": "SNOMED/230690007",
    "Z87": "SNOMED/417662000",
    "Z87891": "SNOMED/266569009",
    "Z95": "SNOMED/428987008",
}

# itemid → in-vocab CLMBR string when mimic-code's exact OMOP id is OOV.
# Parent / acceptable-lossy rollups only (not nursing workflow fields).
ITEM_FALLBACK = {
    220048: "SNOMED/364075005",  # Heart Rhythm; LOINC/8884-9 is OOV
    220739: "LOINC/35088-4",  # GCS eye → GCS panel
    223900: "LOINC/35088-4",
    223901: "LOINC/35088-4",
    229381: "LOINC/35088-4",  # orientation
    228397: "LOINC/35088-4",  # commands response
    228398: "LOINC/35088-4",  # commands
    228414: "LOINC/35088-4",  # speech
    52172: "LOINC/788-0",  # RDW-SD fL → RDW %
    226732: "SNOMED/18286008",  # O2 delivery device → oxygen therapy
    223781: "SNOMED/22253000",  # pain present
    225113: "SNOMED/22253000",
    223783: "SNOMED/22253000",  # pain location
    223784: "SNOMED/22253000",  # pain cause
    223791: "LOINC/38214-3",  # pain 0-10 → VAS pain severity
    224409: "LOINC/38214-3",
    223794: "LOINC/38214-3",
    223792: "SNOMED/225399009",  # pain management → pain assessment
    224031: "SNOMED/267038008",  # edema site → edema
    224027: "LOINC/8310-5",  # skin temp → body temp
    224650: "SNOMED/251175005",  # ectopy type → VPC
    224274: "CPT4/36000",  # peripheral IV gauges
    224275: "CPT4/36000",
    224276: "CPT4/36000",
    224277: "CPT4/36000",
    224566: "CPT4/36000",
    225205: "CPT4/36000",
    225459: "CPT4/71045",  # chest x-ray
    225792: "SNOMED/40617009",  # invasive ventilation
    224385: "SNOMED/112798008",  # intubation
    227194: "SNOMED/112798008",  # extubation → ET tube procedure
    225664: "LOINC/2339-0",  # glucose finger stick
    51484: "LOINC/2514-8",  # urine/serum ketone
    229581: "CPT4/71045",  # portable CXR
    225802: "SNOMED/108241001",  # CRRT → dialysis
    224270: "SNOMED/108241001",  # dialysis catheter
    227712: "CPT4/32551",  # chest tube removed
    226588: "SNOMED/264957007",  # chest tube output
    226589: "SNOMED/264957007",
    229413: "SNOMED/264957007",
    225170: "SNOMED/116859006",  # platelets
    226367: "SNOMED/116859006",  # FFP
}

HCPCS_PHRASE = {
    "hospital observation services": "CPT4/99218",
    "hospital observation per hr": "CPT4/99218",
    "perc drug-el cor stent sing": "CPT4/92928",
    "percutaneous drug eluting coronary stent": "CPT4/92928",
}

MAR_STATUS = {
    "administered",
    "flushed",
    "not given",
    "unk",
    "confirmed",
    "not flushed",
    "started",
    "stopped",
    "paused",
    "held",
    "assessed",
    "delayed",
    "cancelled",
    "canceled",
    "not given per sliding scale",
    "started per sliding scale",
}

_DOSE = re.compile(
    r"\b(\d+\.?\d*\s*(mg|mcg|g|ml|meq|units?|%|iu)|iv|po|im|sc|tab|cap|flush|neb|patch|cream|ointment)\b",
    re.I,
)
_PARENS = re.compile(r"\(.*?\)")
_SPACES = re.compile(r"\s+")
_HCPCS_CODE = re.compile(r"^[A-Z]\d{4}$")
_TRAILING_SALT = re.compile(
    r"\b(hydrochloride|hcl|tartrate|succinate|mesylate|citrate|acetate|"
    r"maleate|phosphate|bitartrate|gluconate|fumarate|besylate|sodium|"
    r"potassium|calcium)\s*$"
)

# Workflow / documentation charting, not vitals, labs, GCS, or pain scores.
# Used only for coverage ("events excluding nursing"), not for mapping.
NURSING_CATEGORIES = frozenset(
    {
        "Care Plans",
        "Restraint/Support Systems",
        "Alarms",
        "Adm History/FHPA",
        "OT Notes",
        "MD Progress Note",
        "Family Mtg Note",
        "Pastoral Care Note",
        "7-Communication",
        "Case Management",
        "PatientSafetyInitialNote",
        "Research Enrollment Note",
        "RNTriggerNote",
        "Triggers Note",
        "Block Charting Note",
    }
)
NURSING_LABEL_RE = re.compile(
    r"head of bed|^turn$|turn q|"
    r"^position$|activity tolerance|^activity$|"
    r"education topic|alarms on|parameters checked|"
    r"less restrictive|safety restraint|\brestraint\b|"
    r"\bncp\b|lung sounds|breath sounds|family meeting|"
    r"^assistance$|pressure reducing|anti.?embolic",
    re.I,
)
# MIMIC labitems that are only H/I/L specimen flags, not analytes.
SPECIMEN_FLAG_ITEMIDS = frozenset({50934, 51678, 50947})


def extract_itemid(code: str) -> Optional[int]:
    if not code:
        return None
    parts = code.split("//")
    if code.startswith("LAB//") or code.startswith("SUBJECT_FLUID_OUTPUT//"):
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return None
    if code.startswith(("PROCEDURE//START//", "PROCEDURE//END//", "INFUSION_START//", "INFUSION_END//")):
        try:
            return int(parts[-1])
        except (IndexError, ValueError):
            return None
    return None


class NursingIndex:
    """Classify MEDS codes as nursing-workflow charting vs clinical events."""

    def __init__(self, itemid_nursing: dict[int, str], label_by_itemid: dict[int, str]):
        self.itemid_nursing = itemid_nursing
        self.label_by_itemid = label_by_itemid

    @classmethod
    def load(cls, *table_paths: str) -> "NursingIndex":
        itemid_nursing: dict[int, str] = {}
        label_by_itemid: dict[int, str] = {}
        for path in table_paths:
            if not path or not os.path.isfile(path):
                continue
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                for row in csv.DictReader(f):
                    raw = (row.get("itemid") or "").strip()
                    if not raw:
                        continue
                    try:
                        itemid = int(raw)
                    except ValueError:
                        continue
                    label = (row.get("label") or "").strip()
                    if label and itemid not in label_by_itemid:
                        label_by_itemid[itemid] = label
                    cat = (row.get("category") or "").strip()
                    if cat in NURSING_CATEGORIES:
                        itemid_nursing[itemid] = cat
        return cls(itemid_nursing, label_by_itemid)

    def is_nursing(self, code: str) -> bool:
        itemid = extract_itemid(code)
        if itemid is not None:
            if itemid in SPECIMEN_FLAG_ITEMIDS or itemid in self.itemid_nursing:
                return True
            label = self.label_by_itemid.get(itemid, "")
            if label and NURSING_LABEL_RE.search(label):
                return True
        return False

    def reason(self, code: str) -> Optional[str]:
        itemid = extract_itemid(code)
        if itemid is None:
            return None
        if itemid in SPECIMEN_FLAG_ITEMIDS:
            return "specimen_flag"
        if itemid in self.itemid_nursing:
            return f"category:{self.itemid_nursing[itemid]}"
        label = self.label_by_itemid.get(itemid, "")
        if label and NURSING_LABEL_RE.search(label):
            return f"label:{label}"
        return None


def fallback_loinc(code: str) -> Optional[str]:
    fb = LOINC_FALLBACK.get((code or "").strip())
    return f"LOINC/{fb}" if fb else None


def loinc_from_label(label: str) -> Optional[str]:
    text = label or ""
    for pat, tok in LABEL_LOINC:
        if pat.search(text):
            return tok
    return None


def normalize_name(text: str) -> str:
    s = text.lower().replace("_", " ")
    s = _PARENS.sub(" ", s)
    s = _DOSE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return _SPACES.sub(" ", s).strip()


def name_variants(text: str) -> list[str]:
    """Ordered lookup keys for a MIMIC drug string."""
    key = normalize_name(text)
    if not key:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    add(key)
    stemmed = _TRAILING_SALT.sub("", key).strip()
    add(stemmed)
    add(key.split(" ")[0])
    for part in re.split(r"[/\-]", key):
        add(part)
        add(_TRAILING_SALT.sub("", part).strip())
    return out


def icd9_proc(raw: str) -> str:
    c = raw.strip().replace(".", "")
    if len(c) <= 2:
        return c
    return f"{c[:2]}.{c[2:]}"


def icd9cm(raw: str) -> str:
    c = raw.strip().replace(".", "").upper()
    if not c:
        return c
    if c[0] in "EV":
        if len(c) <= 3:
            return c
        return f"{c[:3]}.{c[3:]}"
    if len(c) <= 3:
        return c
    return f"{c[:3]}.{c[3:]}"


def icd10cm(raw: str) -> str:
    c = raw.strip().replace(".", "").upper()
    if len(c) <= 3:
        return c
    return f"{c[:3]}.{c[3:]}"


def icd_lookup_keys(raw: str) -> list[str]:
    dotted9 = icd9cm(raw)
    dotted10 = icd10cm(raw)
    undotted = raw.strip().replace(".", "").upper()
    keys = [raw.strip(), undotted, dotted9, dotted10]
    seen: set[str] = set()
    out = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def icd_ancestor_keys(raw: str, version: str) -> list[str]:
    """Exact ICD plus parents (I2510→I251→I25; 41401→4140→414)."""
    undotted = raw.replace(".", "").upper()
    if not undotted:
        return []
    chain = [undotted]
    min_len = 3
    if version == "9" and undotted[:1] in "EV":
        min_len = 3
    for n in range(len(undotted) - 1, min_len - 1, -1):
        chain.append(undotted[:n])
    seen: set[str] = set()
    out: list[str] = []
    for code in chain:
        for key in icd_lookup_keys(code):
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


@dataclass
class MapResult:
    clmbr_code: Optional[str]
    category: str
    stage: str  # in_vocab | oov | no_map


class CodeMapper:
    def __init__(self, vocab: ClmbrVocab, lookups: dict, use_rollups: bool = True):
        self.vocab = vocab
        self.use_rollups = use_rollups
        self.lab_itemid: dict[int, str] = {
            int(k): v for k, v in (lookups.get("lab_itemid") or {}).items()
        }
        self.lab_pair: dict[tuple[int, str], str] = {}
        for k, v in (lookups.get("lab_itemid_unit") or {}).items():
            itemid_s, _, unit = str(k).partition("||")
            self.lab_pair[(int(itemid_s), unit)] = v
        self.lab_label: dict[str, str] = lookups.get("lab_label") or {}
        self.rx_name: dict[str, str] = lookups.get("rxcui_by_name") or {}
        self.infusion: dict[int, str] = {
            int(k): v for k, v in (lookups.get("infusion_itemid") or {}).items()
        }
        self.proc_itemid: dict[int, str] = {
            int(k): v for k, v in (lookups.get("proc_itemid") or {}).items()
        }
        self.output_itemid: dict[int, str] = {
            int(k): v for k, v in (lookups.get("output_itemid") or {}).items()
        }
        self.icd9cm: dict[str, str] = lookups.get("icd9cm") or {}
        self.icd10cm: dict[str, str] = lookups.get("icd10cm") or {}

    @classmethod
    def load(cls, maps_dir: str, vocab: ClmbrVocab, use_rollups: bool = True) -> "CodeMapper":
        path = os.path.join(maps_dir, "lookups.json.gz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} missing. Compile maps with: python -m src.clmbr.build_maps"
            )
        with gzip.open(path, "rt") as f:
            lookups = json.load(f)
        return cls(vocab, lookups, use_rollups=use_rollups)

    def _pick(self, candidates: list[str], category: str) -> MapResult:
        for cand in candidates:
            if cand and self.vocab.contains(cand):
                return MapResult(cand, category, "in_vocab")
        if candidates:
            return MapResult(candidates[0], category, "oov")
        return MapResult(None, category, "no_map")

    def _race_token(self, race: str) -> list[str]:
        raw = race.replace("RACE_", "").strip()
        upper = raw.upper()
        if upper in RACE_TO_CLMBR:
            return [RACE_TO_CLMBR[upper]]
        cands = []
        if "HISPANIC" in upper or "LATINO" in upper:
            cands.append("Ethnicity/Hispanic")
        if "WHITE" in upper:
            cands.append("Race/5")
        elif "BLACK" in upper or "AFRICAN" in upper:
            cands.append("Race/3")
        elif "ASIAN" in upper:
            cands.append("Race/2")
        elif "HAWAII" in upper or "PACIFIC" in upper:
            cands.append("Race/4")
        elif "INDIAN" in upper or "ALASKA" in upper or "NATIVE" in upper:
            cands.append("Race/1")
        return cands

    def _rx_candidates(self, name: str) -> list[str]:
        if not name or normalize_name(name) in MAR_STATUS:
            return []
        cands: list[str] = []
        for key in name_variants(name):
            tok = self.rx_name.get(key)
            if tok and tok not in cands:
                cands.append(tok)
        return cands

    def map_code(self, code: str) -> MapResult:
        if not code:
            return MapResult(None, "other", "no_map")

        if code in NAMED_LOINC:
            return self._pick([NAMED_LOINC[code]], "vital")

        if code in BP_LOINC or code.startswith("Blood Pressure"):
            return self._pick([BP_LOINC.get(code, "")], "blood_pressure")

        if code.startswith("SUBJECT_WEIGHT_AT_INFUSION"):
            return self._pick([WEIGHT_LOINC], "vital")

        if code.startswith("SUBJECT_FLUID_OUTPUT//"):
            parts = code.split("//")
            cands = []
            try:
                itemid = int(parts[1])
            except (IndexError, ValueError):
                itemid = None
            if itemid is not None and itemid in self.output_itemid:
                cands.append(self.output_itemid[itemid])
            if self.use_rollups and itemid is not None:
                fb = ITEM_FALLBACK.get(itemid)
                if fb and fb not in cands:
                    cands.append(fb)
            return self._pick(cands, "output")

        if code.startswith("LAB//"):
            parts = code.split("//")
            cands: list[str] = []
            try:
                itemid = int(parts[1])
            except (IndexError, ValueError):
                itemid = None
            unit = parts[2] if len(parts) > 2 else ""
            if itemid is not None:
                if unit:
                    t = self.lab_pair.get((itemid, unit))
                    if t:
                        cands.append(t)
                t = self.lab_itemid.get(itemid)
                if t and t not in cands:
                    cands.append(t)
                if self.use_rollups:
                    t = ITEM_FALLBACK.get(itemid)
                    if t and t not in cands:
                        cands.append(t)
            return self._pick(cands, "lab")

        if code.startswith("GENDER//"):
            return self._pick([f"Gender/{code.split('//', 1)[1]}"], "gender")

        if code.startswith("RACE//"):
            return self._pick(self._race_token(code.split("//", 1)[1]), "race")

        if code.startswith("HCPCS//"):
            raw = code.split("//", 1)[1].strip()
            cands = []
            if _HCPCS_CODE.match(raw.upper()):
                cands.append(f"HCPCS/{raw.upper()}")
            if self.use_rollups:
                phrase = normalize_name(raw)
                tok = HCPCS_PHRASE.get(phrase)
                if tok:
                    cands.append(tok)
            return self._pick(cands, "hcpcs")

        if code.startswith("PROCEDURE//ICD//"):
            parts = code.split("//")
            cands = []
            if len(parts) >= 4:
                version, raw = parts[2], parts[3]
                if version == "9":
                    cands.append(f"ICD9Proc/{icd9_proc(raw)}")
                    cands.append(f"ICD9Proc/{raw}")
                elif version == "10":
                    cands.append(f"ICD10PCS/{raw}")
                    cands.append(f"ICD10PCS/{raw.upper()}")
            return self._pick(cands, "procedure")

        if code.startswith("PROCEDURE//START//") or code.startswith("PROCEDURE//END//"):
            try:
                itemid = int(code.split("//")[-1])
            except ValueError:
                itemid = None
            cands = []
            if itemid in self.proc_itemid:
                cands.append(self.proc_itemid[itemid])
            if self.use_rollups:
                fb = ITEM_FALLBACK.get(itemid) if itemid is not None else None
                if fb and fb not in cands:
                    cands.append(fb)
            return self._pick(cands, "procedure")

        if code.startswith("DIAGNOSIS//ICD//"):
            parts = code.split("//")
            cands = []
            if len(parts) >= 4:
                version, raw = parts[2], parts[3]
                table = self.icd9cm if version == "9" else self.icd10cm
                builtin = ICD9_SNOMED if version == "9" else ICD10_SNOMED
                keys = (
                    icd_ancestor_keys(raw, version)
                    if self.use_rollups
                    else icd_lookup_keys(raw)
                )
                for key in keys:
                    tok = table.get(key)
                    if tok and tok not in cands:
                        cands.append(tok)
                    u = key.replace(".", "").upper()
                    tok = builtin.get(u) or builtin.get(key)
                    if tok and tok not in cands:
                        cands.append(tok)
            return self._pick(cands, "diagnosis")

        if code.startswith("INFUSION_START//") or code.startswith("INFUSION_END//"):
            try:
                itemid = int(code.split("//")[-1])
            except ValueError:
                itemid = None
            cands = [self.infusion[itemid]] if itemid in self.infusion else []
            if self.use_rollups and itemid is not None:
                fb = ITEM_FALLBACK.get(itemid)
                if fb and fb not in cands:
                    cands.append(fb)
            return self._pick(cands, "medication")

        if code.startswith("MEDICATION//"):
            parts = [p for p in code.split("//") if p]
            name = ""
            if len(parts) >= 3 and parts[1] in {"START", "STOP"}:
                name = parts[2]
            elif len(parts) >= 2:
                name = parts[1]
            return self._pick(self._rx_candidates(name), "medication")

        for prefix, tok in VISIT_RULES:
            if code.startswith(prefix):
                extra = []
                if prefix == "HOSPITAL_ADMISSION":
                    upper = code.upper()
                    if "EW EMER" in upper or "EMERGENCY" in upper or "EU OBSERVATION" in upper:
                        extra = ["Visit/ER", "Visit/ERIP", tok]
                    else:
                        extra = [tok]
                else:
                    extra = [tok]
                return self._pick(extra, "visit_transfer")

        if self.use_rollups and code.startswith("TRANSFER_TO"):
            upper = code.upper()
            if "ED" in upper or "EMERGENCY" in upper:
                return self._pick(["Visit/ER", "Visit/ERIP", "Visit/IP"], "visit_transfer")
            return self._pick(["Visit/IP"], "visit_transfer")

        result = MapResult(None, "other", "no_map")
        if code.startswith(("HOSPITAL_", "ICU_", "ED_", "TRANSFER_")):
            result.category = "visit_transfer"
        return result
