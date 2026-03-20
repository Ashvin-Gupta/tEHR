"""
HDBSCAN Hyperparameter Search + Clinical Cluster Analysis.

Pipeline:
    1. Load held-out embeddings and the corresponding dataset.
    2. Run a 30-trial random search over HDBSCAN hyperparameters (n_neighbors,
       min_cluster_size, min_samples, cluster_selection_method), using a
       10-dimensional UMAP projection as the clustering space.
    3. Score each configuration on cluster purity (case/control separation),
       noise fraction, and whether the cluster count falls in [5, 30].
    4. Re-run the best configuration and produce:
         - UMAP 2D scatter plot coloured by cluster and by case/control status.
         - Clinical fingerprint heatmap: enrichment scores for terms of interest.
         - Cluster composition CSV and per-cluster differential-feature CSVs.
"""

import json
import os
import random
from collections import defaultdict

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import yaml

from src.data.dataset_factory import get_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = "src/configs/classification_config.yaml"
EMBEDDINGS_DIR = (
    "/gpfs/scratch/qc25022/pancreas/experiments/Results/"
    "lora-1-month-logistic-refactored-new-time-mistral-7B/"
    "patient_clustering_heldout"
)
OUTPUT_DIR = EMBEDDINGS_DIR + "/hdbscan_search"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_TRIALS = 30
RANDOM_SEED = 42

MY_TERMS = [
    "Secondary polycythaemia", "ADHD", "Benign prostatic hyperplasia",
    "Bipolar affective disorder", "Bisphosphonates", "Black", "COPD",
    "Cervical intraepithelial neoplasia", "Chronic cystitis", "Chronic hepatitis",
    "Chronic kidney disease", "Cleidocranial dysplasia", "Coronary heart disease",
    "Covid2", "Diabetes", "Drinker - excess/disorder", "Drinker - unspecified",
    "Drinker - within limits", "End-Stage renal disease", "Epsstein-Barr virus",
    "Gastro-oesophageal reflux", "Giant cell arteritis", "Glomeruonephritis",
    "High birth weight", "HPV", "IBD", "IBS", "IBS medication", "Ischaemic stroke",
    "Left bundle branch block", "Low birth weight", "Myelodysplastic syndrome",
    "Monoclonal gammopathy undetermined significance", "MND", "Non-Hodgkin lymphoma",
    "Non-Alcoholic fatty liver disease", "Non-drinker", "Oesteoarthritis",
    "Postcoital and contact bleeding", "PCOS", "Polycythaemia vera",
    "Patent ductus arteriousus", "Pulmonary embolism", "Postmenopausal bleeding",
    "Polymyalgia rheumatica", "Psoriatic arthropathy", "PTSD",
    "Personality disorder", "Right bundle branch block",
    "Respiratory distress of newborn", "Rheumaticn valve disease",
    "Rheumatoid arthritis", "Lupus erythematosus", "Supraventricular tachycardia",
    "Stroke NOS", "Subarachnoid haemorrhage", "Tuberculosis",
    "Transient ischaemic attack", "Tubulo-interstitial nephritis",
    "Ventricular tachycardia", "Venous thromboembolism", "Abdominal mass",
    "Achalasia", "Acne", "Actinic keratosis", "Adrenal insufficiency",
    "Agranulocytosis", "Alcohol problems", "Alcohol Dependency", "Allergic rhinitis",
    "Alopecia areata", "Amylase", "Anal fissure", "Angiodysplasia colon",
    "Ankylosing spondylitus", "Anorectal fistula", "Anorectal prolapse",
    "Anterior uveitis", "Anxiety or phobia", "Aortic aneurysm", "Aplastic",
    "Appendicitis", "Appetite loss", "Arthritis", "Asbestosis",
    "Aspiration pneumonia", "Asthma", "Atrial fibrilation", "Autism",
    "Autoimmune liver disease", "Autoimmune skin disease",
    "Autonomic nervous system", "Atrioventricular block first degree",
    "Atrioventricular block second degree", "Atrioventricular block third degree",
    "Back pain", "Bariatric surgery", "Barretts", "Bells's palsy",
    "Benign brain neoplasm", "Benign colon neoplasm", "Benign ovary neoplasm",
    "Benign stomach neoplasm", "Benign uterus neoplasm", "Bifascicular block",
    "Biliary cancer", "Bilirubin", "Bladder cancer", "Blindness",
    "Bloating distension", "Blood pressure", "BMI", "Bone cancer", "Diastolic",
    "Systolic", "Brain cancer", "Breast cancer", "Bronchiectasis",
    "Cardiomyopathy", "Carpal tunnel", "Cataract", "Cerebral palsy",
    "Cerebrovascular disease", "Cervix cancer", "Change in bowel habit",
    "Chest pain", "Cholangitis", "Cholecystectomy", "Cholecystitis",
    "Cholelithiasis", "Cholesterol triglycerides", "Cholesterolaemia",
    "Chronic fatigue", "Carcinoma in cervical", "Cirrhosis", "Coeliac",
    "Collapsed vertabrae", "Colorectal cancer", "Congenital cardiac",
    "Constipation", "Creatinine serum", "Creatinine urine", "Crohn's",
    "Current or ex-smoker", "Current smoker", "Cystic fibrosis", "Cystic renal",
    "Dilated cardiomyopathy", "Deaf", "Delirium", "Dementia", "Depression",
    "Dermatitis", "Diabetes eye", "Diabetes neuro", "Diabetes type1",
    "Diabetes type2", "Diabetic retinopathy", "Diarrhoea", "Diverticular",
    "Diabetic neurological complications", "Down syndrome", "Dysmenorrhoea",
    "Dyspepsia", "Dysphagia", "Eating disorder", "Endometrial hyper",
    "Endometriosis", "Endoscopy", "Enteropathic arthropathy", "Enthesopathy",
    "Epigastric pain", "Epilepsy", "Erectile dysfunction", "Esophagus cancer",
    "Essential tremor", "Ex-smoker", "Family history of colon cancer",
    "Family history of esophagus cancer", "Family history of git cancer",
    "Family history of liver cancer", "Family history of other cancer",
    "Family history of pancreas cancer", "Family history of stomach cancer",
    "Fatigue malaise", "Fatty liver", "Female genital prolapse",
    "Female infertility", "Ferritin", "Fibromatosis", "Fibromyalgia",
    "Folate deficiency", "Fracture hip", "Fracture wrist", "Gallbladder disease",
    "Gallbladder disease without PSC", "Gastrectomy", "Gastritis duodenitis",
    "Gastrointestinal angiodysplasia", "GIT bleed", "Glaucoma", "Glucose",
    "GORD PUDB Barrets", "Gout", "Haemangioma", "Hayfever", "Headneck cancer",
    "Hearing loss", "Heart failure", "Heart valve dz", "Heartburn", "Height",
    "Hep B vaccine", "Hepatitis B", "Hepatitis C", "Hernia abdominal",
    "Hernia diaphragm", "Hidradenitis", "HIV", "Hoarseness",
    "Hypertrophic cardiomyopathy", "Hodgkins", "Hydrocele", "Hyperparathyroidism",
    "Hyper nasal turbs", "Hyperprolactinaemia", "Hypertension", "Hypopituitarism",
    "Hyposplenism", "Immunodef", "Infection bones joints",
    "Intellectual disability", "Intervertebral disc disorders",
    "Intracerebral haemorrhage", "Intracranial hypertension",
    "Intrauterine hypoxia", "Iron", "Iron medication", "Iron deficiency anaemia",
    "Jaundice", "Juvenile arthritis", "Keratitis", "Kidney cancer", "Leiomyoma",
    "Leukemia cancer", "Lichen planus", "Liver cancer", "Alcoholic liver disease",
    "Liver disease", "Liver disease with autoimmune", "Liver failure",
    "Lung cancer", "Lymphocyte", "Lymphoma cancer", "Macular degeneration",
    "Male infertility", "Melanoma cancer", "Meniere", "Menorrhagia",
    "Mesothelioma cancer", "Migraine", "Multiple valve diseas",
    "Multiple sclerosis", "Myasthenia", "Myeloma cancer", "Myocardial infarction",
    "Nasal polyp", "Nausea vomiting", "Neonatal jaundice", "Neuropathic bladder",
    "Neutrophils", "Nicotine or tobacco use", "Non-smoker",
    "Nonrheumatic aortic valve disorder", "Nonrheumatic mitral valve disorder",
    "Obesity", "Obstructive reflux uropathy", "OCD", "Odynophagia",
    "Oesophagus ulcer", "Oral cancer", "Osteoporosis", "Other cancer",
    "Other anaemia", "Other haem anaemia", "Ovary cancer", "Pancreas cancer",
    "Pancreatitis", "Parkinsons", "Pericardial effusion",
    "Peripheral atrerial disease", "Peripheral neuropathy", "Peritonitis",
    "Phlebitis", "Pilonidal", "Plasma proteins", "Plasma viscosity",
    "Plasmacell", "Platelets", "Pleural effusion", "Pleural plaque", "Pneumonia",
    "Pneumothorax", "Polyposis", "Portal hypertension", "Post term infant",
    "Posterior uveitis", "Prematurity", "Prostate cancer", "Pruritus",
    "Psoriasis", "Ptosis", "Pulmonary collapse", "Pulmonary fibrosis",
    "Pulmonary hypertension", "raynauds", "Reactive anthropathies",
    "Respiratory failure", "Retinal detachment", "Retinal vasc occlussion",
    "Retinal vein occlusion", "Rheumatic fever", "Rosacea", "Sarcoid",
    "Schizophrenia", "Scleritis", "Scoliosis", "Seborrhoeic dermatitis",
    "Sepsis newborn", "Sick sinus", "Sickle cell", "Sickle trait", "Sinusitis",
    "Sjogrens disease", "Skin ulcer", "Sleep apnoea", "Sleep disturbance",
    "Somatoform", "Spina bifida", "Spinal stenosis", "Splenomegaly",
    "Spondylolisthesis", "Spondylosis", "Stable angina", "Steatorrhea",
    "Stomach cancer", "Subarach", "Subdural haematoma", "Substance misuse",
    "Systemic sclerosis", "Testis cancer", "Thalassaemia", "Thrombocytopaenia",
    "Thrombophilia", "Thyroid", "Tinnitus", "Trifascicular block",
    "Trigeminal neuralgia", "Ulcerative colitis", "Peptic ulcer disease",
    "Undescended testis", "Unspecified rare diabetes", "Unstable angina",
    "Uppergit cancer", "Urea serum", "Urinary incontinence", "Urolithiasis",
    "Urticaria", "Uterine cancer", "Varices portal hypertension",
    "Venous lymphatic", "Venous thromboembolism", "Vitb12 deficiency",
    "Vitiligo", "Volvulus", "Weight", "Weightloss",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(output_dir: str):
    embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
    patient_ids_path = os.path.join(output_dir, "patient_ids.npy")
    if os.path.exists(patient_ids_path):
        patient_ids = np.load(patient_ids_path, allow_pickle=True).tolist()
    else:
        patient_ids = list(range(embeddings.shape[0]))
    return embeddings, patient_ids


def load_dataset(config_path: str, split: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    dataset_name = data_config.get("dataset", "CPRD")

    dataset = get_dataset(dataset_name, data_config, split)

    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    label_to_name = {v: k for k, v in dataset.label_to_id.items()}

    subject_to_record_idx = {}
    for idx, record in enumerate(dataset.patient_records):
        sid = record.get("subject_id")
        if sid is not None:
            subject_to_record_idx[str(sid)] = idx

    return dataset, subject_to_label, label_to_name, subject_to_record_idx


# ---------------------------------------------------------------------------
# Token filtering
# ---------------------------------------------------------------------------
def get_filtered_token_ids(dataset, record: dict) -> list:
    """Return token IDs after applying time cutoff and removing cancer tokens."""
    subject_id = record.get("subject_id")
    token_ids = list(record.get("tokens", []))
    timestamps = record.get("timestamps", [])

    if subject_id is None:
        return token_ids

    label = dataset.subject_to_label.get(subject_id)
    token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, label)

    return [
        tid for tid in token_ids
        if "cancer" not in str(dataset.id_to_token.get(tid, "")).lower()
    ]


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------
def reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float = 0.0,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Reduce embeddings to n_components dimensions using UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def calculate_purity_score(df_clusters: pd.DataFrame) -> float:
    """
    Score how well clusters separate cases from controls.

    - avg_purity: mean of |case_rate - 0.5| * 2 per cluster (1 = pure, 0 = 50/50 mix).
    - noise_penalty: 1 - noise_fraction (penalises configurations that label many
      patients as noise).
    - cluster_penalty: 0.5 if cluster count is outside [5, 30], else 1.0.

    Final score = avg_purity * noise_penalty * cluster_penalty.
    """
    df_clean = df_clusters[df_clusters["cluster"] != -1]
    if df_clean.empty or len(df_clean["cluster"].unique()) < 2:
        return 0.0

    purity_per_cluster = []
    for cid in df_clean["cluster"].unique():
        subset = df_clean[df_clean["cluster"] == cid]
        case_rate = subset["is_case"].mean()
        purity_per_cluster.append(abs(case_rate - 0.5) * 2)

    avg_purity = float(np.mean(purity_per_cluster))
    noise_ratio = float((df_clusters["cluster"] == -1).mean())
    noise_penalty = 1.0 - noise_ratio

    n_clusters = len(df_clean["cluster"].unique())
    cluster_penalty = 1.0 if 5 <= n_clusters <= 30 else 0.5

    return avg_purity * noise_penalty * cluster_penalty


# ---------------------------------------------------------------------------
# Random hyperparameter search
# ---------------------------------------------------------------------------
def run_random_search(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    n_trials: int = N_TRIALS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly sample HDBSCAN hyperparameter combinations and score each one.

    Search space:
        n_neighbors              : int in [10, 100]  (for the 10D UMAP reduction)
        min_cluster_size         : int in [10, 150]
        min_samples              : int in [1, 30]
        cluster_selection_method : 'eom' or 'leaf'

    10D UMAP reductions are cached by n_neighbors to avoid redundant computation.
    Returns a DataFrame of all trial results sorted by score descending.
    """
    rng = random.Random(seed)
    umap_cache: dict = {}
    results = []

    for trial in range(1, n_trials + 1):
        n_neighbors = rng.randint(10, 100)
        min_cluster_size = rng.randint(10, 150)
        min_samples = rng.randint(1, 30)
        method = rng.choice(["eom", "leaf"])

        if n_neighbors not in umap_cache:
            print(f"  [Trial {trial}/{n_trials}] UMAP 10D (n_neighbors={n_neighbors})...")
            umap_cache[n_neighbors] = reduce_umap(
                embeddings, n_components=10, n_neighbors=n_neighbors
            )

        emb_10d = umap_cache[n_neighbors]

        cluster_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
            prediction_data=False,
        ).fit_predict(emb_10d)

        temp_df = pd.DataFrame({"cluster": cluster_labels, "is_case": binary_labels})
        score = calculate_purity_score(temp_df)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_pct = float((cluster_labels == -1).mean() * 100)

        results.append({
            "trial": trial,
            "n_neighbors": n_neighbors,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": method,
            "score": round(score, 4),
            "n_clusters": n_clusters,
            "noise_pct": round(noise_pct, 2),
        })
        print(
            f"  [Trial {trial}/{n_trials}] "
            f"nn={n_neighbors}, mcs={min_cluster_size}, ms={min_samples}, "
            f"method={method} → score={score:.3f}, clusters={n_clusters}, "
            f"noise={noise_pct:.1f}%"
        )

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Apply best params
# ---------------------------------------------------------------------------
def apply_best_params(
    embeddings: np.ndarray,
    best: dict,
) -> tuple:
    """
    Re-run the winning 10D UMAP + HDBSCAN and produce a separate 2D UMAP
    projection for visualisation.

    Returns:
        cluster_labels : 1D int array
        z_2d           : (N, 2) float array
    """
    print(f"\nApplying best params: {best}")

    print("  Running 10D UMAP for clustering...")
    emb_10d = reduce_umap(
        embeddings, n_components=10, n_neighbors=int(best["n_neighbors"])
    )

    print("  Running HDBSCAN...")
    cluster_labels = hdbscan.HDBSCAN(
        min_cluster_size=int(best["min_cluster_size"]),
        min_samples=int(best["min_samples"]),
        cluster_selection_method=best["cluster_selection_method"],
        prediction_data=True,
    ).fit_predict(emb_10d)

    print("  Running 2D UMAP for visualisation...")
    z_2d = reduce_umap(
        embeddings, n_components=2, n_neighbors=15, min_dist=0.1
    )

    return cluster_labels, z_2d


# ---------------------------------------------------------------------------
# Cluster composition
# ---------------------------------------------------------------------------
def calculate_cluster_composition(df_clusters: pd.DataFrame, output_dir: str):
    df_clusters["binary_label"] = df_clusters["case_label"].apply(
        lambda x: "Control" if str(x).lower() == "control" else "Case"
    )
    composition = (
        df_clusters.groupby("cluster")["binary_label"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in ["Case", "Control"]:
        if col not in composition.columns:
            composition[col] = 0
    composition["Total"] = composition["Case"] + composition["Control"]
    composition["Case_%"] = (composition["Case"] / composition["Total"] * 100).round(2)

    print("\n--- Cluster Composition ---")
    print(composition.sort_values("Total", ascending=False))

    composition.to_csv(os.path.join(output_dir, "cluster_case_control_composition.csv"))
    return composition


# ---------------------------------------------------------------------------
# Differential feature analysis
# ---------------------------------------------------------------------------
def analyse_differential_features(
    dataset,
    df_clusters: pd.DataFrame,
    subject_to_record_idx: dict,
    output_dir: str,
):
    """
    For every LAB/MEASUREMENT/MEDICAL token compute an enrichment score per
    cluster: (frequency in cluster) / (frequency globally).  >1 means the
    event is over-represented in that cluster.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    for _, row in df_clusters.iterrows():
        pid = str(row["patient_id"])
        cluster = int(row["cluster"])
        rec_idx = subject_to_record_idx.get(pid)
        if rec_idx is None:
            continue

        record = dataset.patient_records[rec_idx]
        token_ids = get_filtered_token_ids(dataset, record)
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in token_ids]

        for i, code in enumerate(tokens):
            is_measurement = any(
                code.startswith(p)
                for p in ["LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"]
            )
            is_medical = code.startswith("MEDICAL//")

            if is_measurement or is_medical:
                val = np.nan
                if is_measurement and i + 1 < len(tokens):
                    try:
                        val = float(tokens[i + 1])
                    except ValueError:
                        pass
                all_data.append({"cluster": cluster, "event_code": code, "value": val})

    df_raw = pd.DataFrame(all_data)

    global_counts = df_raw["event_code"].value_counts()
    global_medians = df_raw.groupby("event_code")["value"].median()
    total_patients = df_clusters["patient_id"].nunique()

    summary_rows = []
    for cluster_id in sorted(df_clusters["cluster"].unique()):
        cluster_df = df_raw[df_raw["cluster"] == cluster_id]
        cluster_size = (
            df_clusters[df_clusters["cluster"] == cluster_id]["patient_id"].nunique()
        )

        for event_code, count in cluster_df["event_code"].value_counts().items():
            vals = cluster_df[cluster_df["event_code"] == event_code]["value"].dropna()

            global_freq = global_counts[event_code] / total_patients
            cluster_freq = count / cluster_size
            enrichment_factor = cluster_freq / global_freq if global_freq > 0 else 0

            mean_v = vals.mean() if not vals.empty else np.nan
            median_v = vals.median() if not vals.empty else np.nan
            diff_from_global = (
                median_v - global_medians[event_code]
                if not np.isnan(median_v)
                else 0
            )

            try:
                human = dataset.translator.translate(event_code).rstrip("; ")
            except Exception:
                human = event_code

            summary_rows.append({
                "cluster": cluster_id,
                "event_code": event_code,
                "event_human": human,
                "count": count,
                "enrichment_score": round(enrichment_factor, 3),
                "mean_value": mean_v,
                "median_value": median_v,
                "diff_from_global_median": diff_from_global,
                "q25_value": vals.quantile(0.25) if not vals.empty else np.nan,
                "q75_value": vals.quantile(0.75) if not vals.empty else np.nan,
            })

    df_final = (
        pd.DataFrame(summary_rows)
        .sort_values(["cluster", "enrichment_score"], ascending=[True, False])
    )
    df_final.to_csv(
        os.path.join(output_dir, "differential_cluster_analysis.csv"), index=False
    )
    for cid in df_clusters["cluster"].unique():
        df_final[df_final["cluster"] == cid].to_csv(
            os.path.join(output_dir, f"cluster_{cid}_analysis.csv"), index=False
        )

    print(f"Differential analysis complete. Files saved to {output_dir}")
    return df_final


# ---------------------------------------------------------------------------
# Clinical reports
# ---------------------------------------------------------------------------
def generate_clinical_reports(
    df_summary: pd.DataFrame, terms_list: list, output_dir: str
):
    """
    Produces:
    1. global_cluster_signatures.csv  — top-20 enriched codes (count > 5) per cluster.
    2. terms_of_interest_pivot.csv    — enrichment scores for MY_TERMS per cluster.

    Returns (pivot_display_df, heatmap_numeric_df) for downstream plotting.
    """
    # --- Global top-20 signatures ---
    global_report = []
    for cid in sorted(df_summary["cluster"].unique()):
        top = (
            df_summary[
                (df_summary["cluster"] == cid) & (df_summary["count"] > 5)
            ]
            .sort_values("enrichment_score", ascending=False)
            .head(20)
        )
        global_report.append(top)
    if global_report:
        pd.concat(global_report).to_csv(
            os.path.join(output_dir, "global_cluster_signatures.csv"), index=False
        )

    # --- Terms of interest matrix ---
    rows = []
    for term in list(dict.fromkeys(terms_list)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if match.empty:
            continue
        match = match.sort_values("count", ascending=False).drop_duplicates(
            subset=["cluster"]
        )
        rows.append(match)

    if not rows:
        return None, None

    df_filtered = pd.concat(rows)

    # Guard against two different raw codes translating to the same human label:
    # keep the highest-count row per (event_human, cluster) pair.
    df_filtered = (
        df_filtered
        .sort_values("count", ascending=False)
        .drop_duplicates(subset=["event_human", "cluster"])
    )

    df_filtered["display"] = df_filtered.apply(
        lambda r: (
            f"{r['median_value']:.1f}"
            if pd.notnull(r["median_value"]) and r["median_value"] != 0
            else f"x{r['enrichment_score']:.1f}"
        ),
        axis=1,
    )

    heatmap_data = df_filtered.pivot(
        index="event_human", columns="cluster", values="enrichment_score"
    ).fillna(0)

    # Sort rows by peak enrichment across clusters so the most discriminating
    # terms appear at the top rather than in alphabetical order.
    row_order = heatmap_data.max(axis=1).sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[row_order]

    pivot = df_filtered.pivot(
        index="event_human", columns="cluster", values="display"
    ).fillna("-").loc[row_order]
    pivot.to_csv(os.path.join(output_dir, "terms_of_interest_pivot.csv"))

    return pivot, heatmap_data


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def visualize_clusters(df_clusters: pd.DataFrame, output_dir: str):
    sns.set(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df_clusters["plot_label"] = df_clusters["case_label"].apply(
        lambda x: "Control" if str(x).lower() == "control" else "Case"
    )

    sns.scatterplot(
        data=df_clusters,
        x="dim1",
        y="dim2",
        hue="cluster",
        palette="viridis",
        s=10,
        alpha=0.6,
        ax=axes[0],
        legend="full",
    )
    axes[0].set_title("UMAP: Clusters")
    axes[0].legend(
        title="Cluster",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        ncol=2 if len(df_clusters["cluster"].unique()) > 20 else 1,
    )

    sns.scatterplot(
        data=df_clusters,
        x="dim1",
        y="dim2",
        hue="plot_label",
        palette={"Control": "lightgrey", "Case": "red"},
        s=10,
        alpha=0.7,
        ax=axes[1],
    )
    axes[1].set_title("UMAP: Cases (Red) vs Controls (Grey)")
    axes[1].legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "umap_clusters_visualization.png"), dpi=300
    )
    plt.close()


def plot_clinical_heatmap(heatmap_data: pd.DataFrame, output_dir: str):
    if heatmap_data is None or heatmap_data.empty:
        return

    if -1 in heatmap_data.columns:
        heatmap_data = heatmap_data.drop(columns=[-1])

    plt.figure(figsize=(14, max(10, len(heatmap_data) * 0.25)))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Enrichment Score"},
    )
    plt.title("Clinical Fingerprint: Enrichment by Cluster")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "clinical_fingerprint_heatmap.png"), dpi=300
    )
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    print("Loading dataset and embeddings...")
    dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(
        CONFIG_PATH, "held_out"
    )
    embeddings, patient_ids = load_embeddings(EMBEDDINGS_DIR)

    binary_labels = np.array(
        [1 if int(subject_to_label.get(str(pid), 0)) > 0 else 0 for pid in patient_ids]
    )

    # 2. Random hyperparameter search
    print(f"\nStarting {N_TRIALS}-trial random search...")
    df_search = run_random_search(embeddings, binary_labels, n_trials=N_TRIALS)

    search_path = os.path.join(OUTPUT_DIR, "hdbscan_search_results.csv")
    df_search.to_csv(search_path, index=False)
    print(f"\nSearch complete. Results saved to {search_path}")
    print("\nTop 5 configurations:")
    print(df_search.head(5).to_string(index=False))

    best = df_search.iloc[0].to_dict()
    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # 3. Apply best configuration
    cluster_labels, z_2d = apply_best_params(embeddings, best)

    # 4. Build cluster DataFrame
    df_clusters = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster": cluster_labels,
        "dim1": z_2d[:, 0],
        "dim2": z_2d[:, 1],
    })
    df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
    df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
    df_clusters["label_name"] = df_clusters["label_id"].map(label_to_name)
    df_clusters["is_case"] = df_clusters["label_id"].fillna(0).astype(int) > 0
    df_clusters["case_label"] = np.where(df_clusters["is_case"], "Case", "Control")

    df_clusters.to_csv(os.path.join(OUTPUT_DIR, "patient_clusters.csv"), index=False)

    # 5. Cluster composition
    calculate_cluster_composition(df_clusters, OUTPUT_DIR)

    # 6. Differential feature analysis
    print("\nRunning differential feature analysis...")
    df_summary = analyse_differential_features(
        dataset, df_clusters, subject_to_record_idx, OUTPUT_DIR
    )

    # 7. Clinical reports and heatmap data
    print("Generating clinical reports...")
    _, heatmap_data = generate_clinical_reports(df_summary, MY_TERMS, OUTPUT_DIR)

    # 8. Figures
    print("Producing figures...")
    visualize_clusters(df_clusters, OUTPUT_DIR)
    plot_clinical_heatmap(heatmap_data, OUTPUT_DIR)

    print(f"\nAll done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
