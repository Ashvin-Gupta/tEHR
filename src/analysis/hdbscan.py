"""
HDBSCAN Clustering & Clinical Phenotyping Pipeline for Medical Journals.

Features:
    - Uses fixed, predefined hyperparameters for reproducibility.
    - Separates Categorical (Enrichment) and Continuous (Median/IQR) clinical markers.
    - Generates publication-ready figures (Annotated UMAP, Z-Score Lab Heatmaps, Composition Bars).
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import joblib

from src.data.dataset_factory import get_dataset

# ---------------------------------------------------------------------------
# Configuration & Hyperparameters (Fixed for Reproducibility)
# ---------------------------------------------------------------------------
CONFIG_PATH = "src/configs/classification_config.yaml"
EMBEDDINGS_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"
OUTPUT_DIR = EMBEDDINGS_DIR + "/journal_clustering_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# Fixed HDBSCAN parameters from previous search
N_NEIGHBORS = 41
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5
CLUSTER_SELECTION_METHOD = "eom"

# --- Categorical Terms (Conditions/Events) ---
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
    "Vitiligo", "Volvulus", "Weight", "Weightloss"
]

# --- Continuous Terms (Lab/Vitals) ---
LAB_TERMS = [
    "Lab MCV", "Lab CRP", "Lab Hemoglobin", "Lab TIBC", "Lab HbA1c", 
    "Lab Plasma viscosity", "Lab ESR", "Lab GGT", "Lab Lymphocytes", 
    "Lab Platelets", "Lab AST", "Lab ALP", "Lab Ferritin", "Lab MCH", 
    "Lab Calcium serum", "Lab Neutrophils", "Lab H_pylori", "Lab Glucose", 
    "Lab Cholesterol triglycerides", "Lab Bilirubin", "Lab Anti-ttg", 
    "Lab Plasma proteins", "Lab Blood pressure", "Lab Amylase", "Lab ALT", 
    "Lab Urea serum", "Lab CA125", "Lab Creatinine serum", "Lab Albumin serum", 
    "Lab WBC", "Lab Creatinine urine", "Lab Iron"
]

# ---------------------------------------------------------------------------
# Data Loading & Base Functions
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
    dataset = get_dataset(config["data"].get("dataset", "CPRD"), config["data"], split)
    
    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    label_to_name = {v: k for k, v in dataset.label_to_id.items()}
    subject_to_record_idx = {str(r.get("subject_id")): idx for idx, r in enumerate(dataset.patient_records) if r.get("subject_id")}
    return dataset, subject_to_label, label_to_name, subject_to_record_idx

def get_filtered_token_ids(dataset, record: dict) -> list:
    subject_id = record.get("subject_id")
    token_ids = list(record.get("tokens", []))
    timestamps = record.get("timestamps", [])
    if subject_id is None: return token_ids
    
    label = dataset.subject_to_label.get(subject_id)
    token_ids = dataset._apply_time_cutoff(token_ids, timestamps, subject_id, label)
    return [tid for tid in token_ids if "cancer" not in str(dataset.id_to_token.get(tid, "")).lower()]

# ---------------------------------------------------------------------------
# Clustering Pipeline
# ---------------------------------------------------------------------------
def run_fixed_clustering(embeddings: np.ndarray) -> tuple:
    print(f"Running clustering with fixed params: nn={N_NEIGHBORS}, mcs={MIN_CLUSTER_SIZE}, ms={MIN_SAMPLES}")
    
    # 1. 10D Reduction
    reducer_10d = umap.UMAP(n_components=10, n_neighbors=N_NEIGHBORS, min_dist=0.0, random_state=RANDOM_SEED, n_jobs=1)
    emb_10d = reducer_10d.fit_transform(embeddings)

    # 2. HDBSCAN
    cluster_labels = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    ).fit_predict(emb_10d)

    # 3. 2D Reduction for visualization
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED, n_jobs=1)
    z_2d = reducer_2d.fit_transform(embeddings)

    return cluster_labels,z_2d, reducer_10d, reducer_2d

# ---------------------------------------------------------------------------
# Data Analysis
# ---------------------------------------------------------------------------
def analyse_differential_features(dataset, df_clusters: pd.DataFrame, subject_to_record_idx: dict, output_dir: str):
    all_data = []
    
    for _, row in df_clusters.iterrows():
        pid, cluster = str(row["patient_id"]), int(row["cluster"])
        if (rec_idx := subject_to_record_idx.get(pid)) is None: continue

        record = dataset.patient_records[rec_idx]
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in get_filtered_token_ids(dataset, record)]

        for i, code in enumerate(tokens):
            is_measurement = any(code.startswith(p) for p in ["LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"])
            is_medical = code.startswith("MEDICAL//")

            if is_measurement or is_medical:
                val = np.nan
                if is_measurement and i + 1 < len(tokens):
                    try: val = float(tokens[i + 1])
                    except ValueError: pass
                all_data.append({"cluster": cluster, "event_code": code, "value": val})

    df_raw = pd.DataFrame(all_data)
    global_counts = df_raw["event_code"].value_counts()
    global_medians = df_raw.groupby("event_code")["value"].median()
    total_patients = df_clusters["patient_id"].nunique()

    summary_rows = []
    for cluster_id in sorted(df_clusters["cluster"].unique()):
        cluster_df = df_raw[df_raw["cluster"] == cluster_id]
        cluster_size = df_clusters[df_clusters["cluster"] == cluster_id]["patient_id"].nunique()

        for event_code, count in cluster_df["event_code"].value_counts().items():
            vals = cluster_df[cluster_df["event_code"] == event_code]["value"].dropna()
            
            # Translated Human Name
            try: human = dataset.translator.translate(event_code).rstrip("; ")
            except Exception: human = event_code

            summary_rows.append({
                "cluster": cluster_id,
                "event_code": event_code,
                "event_human": human,
                "count": count,
                "enrichment_score": round((count / cluster_size) / (global_counts[event_code] / total_patients), 3),
                "median_value": vals.median() if not vals.empty else np.nan,
                "q25_value": vals.quantile(0.25) if not vals.empty else np.nan,
                "q75_value": vals.quantile(0.75) if not vals.empty else np.nan,
                "global_median_diff": vals.median() - global_medians[event_code] if not vals.empty else np.nan,
            })

    df_final = pd.DataFrame(summary_rows)
    df_final.to_csv(os.path.join(output_dir, "full_differential_analysis.csv"), index=False)
    return df_final

def generate_journal_reports(df_summary: pd.DataFrame, output_dir: str):
    """Separates and formats reports for Categorical vs Continuous terms."""
    
    # 1. Categorical (Diseases/Conditions) - Report Fold Enrichment
    cat_rows = []
    for term in list(dict.fromkeys(MY_TERMS)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if not match.empty:
            match = match.sort_values("count", ascending=False).drop_duplicates(subset=["cluster"])
            cat_rows.append(match)
            
    if cat_rows:
        df_cat = pd.concat(cat_rows).drop_duplicates(subset=["event_human", "cluster"])
        df_cat["display"] = df_cat["enrichment_score"].apply(lambda x: f"x{x:.1f}")

        heatmap_cat = df_cat.pivot(index="event_human", columns="cluster", values="enrichment_score").fillna(0)
        # Drop noise cluster before ordering
        if -1 in heatmap_cat.columns:
            heatmap_cat = heatmap_cat.drop(columns=[-1])
        row_order = cluster_priority_row_order(heatmap_cat)
        heatmap_cat = heatmap_cat.loc[row_order]

        cat_pivot = df_cat.pivot(index="event_human", columns="cluster", values="display").fillna("-").loc[row_order]
        cat_pivot.to_csv(os.path.join(output_dir, "table_categorical_enrichment.csv"))
    else:
        heatmap_cat = None

    # 2. Continuous (Labs) - Report Medians & IQR
    lab_rows = []
    for term in list(dict.fromkeys(LAB_TERMS)):
        match = df_summary[df_summary["event_human"] == term].copy()
        if not match.empty:
            match = match.sort_values("count", ascending=False).drop_duplicates(subset=["cluster"])
            lab_rows.append(match)
            
    if lab_rows:
        df_lab = pd.concat(lab_rows).drop_duplicates(subset=["event_human", "cluster"])
        df_lab["display"] = df_lab.apply(
            lambda r: f"{r['median_value']:.1f} [{r['q25_value']:.1f}-{r['q75_value']:.1f}]"
            if pd.notnull(r['median_value']) else "-", axis=1
        )

        # Raw median pivot for Z-score plotting; order rows by cluster priority
        heatmap_lab = df_lab.pivot(index="event_human", columns="cluster", values="median_value")
        if -1 in heatmap_lab.columns:
            heatmap_lab = heatmap_lab.drop(columns=[-1])
        heatmap_lab = heatmap_lab.dropna(how='all')
        if not heatmap_lab.empty:
            # Use filled version only for ordering, keep NaNs for plotting
            filled = heatmap_lab.fillna(0)
            row_order_lab = cluster_priority_row_order(filled)
            heatmap_lab = heatmap_lab.loc[row_order_lab]

        lab_pivot = df_lab.pivot(index="event_human", columns="cluster", values="display").fillna("-")
        if -1 in lab_pivot.columns:
            lab_pivot = lab_pivot.drop(columns=[-1])
        lab_pivot.to_csv(os.path.join(output_dir, "table_continuous_labs_median_iqr.csv"))
    else:
        heatmap_lab = None

    return heatmap_cat, heatmap_lab

# ---------------------------------------------------------------------------
# Row ordering helper
# ---------------------------------------------------------------------------
def cluster_priority_row_order(heatmap_data: pd.DataFrame) -> list:
    """
    Order rows cluster-by-cluster.

    For each cluster (ascending ID order), find the rows where that cluster
    has the highest enrichment value and they haven't already been assigned
    to a prior cluster.  Within each cluster's group, rows are sorted by
    that cluster's value descending.  Any rows not dominated by any single
    cluster are appended at the end.
    """
    ordered = []
    seen = set()

    for col in sorted(heatmap_data.columns):
        remaining_idx = [r for r in heatmap_data.index if r not in seen]
        if not remaining_idx:
            break
        remaining = heatmap_data.loc[remaining_idx]
        dominated = remaining[remaining.idxmax(axis=1) == col]
        dominated = dominated.sort_values(col, ascending=False)
        ordered.extend(dominated.index.tolist())
        seen.update(dominated.index.tolist())

    # Any rows not dominated by any cluster (all-zero or tied)
    ordered.extend([r for r in heatmap_data.index if r not in seen])
    return ordered


# ---------------------------------------------------------------------------
# Journal-Ready Visualizations
# ---------------------------------------------------------------------------
def plot_annotated_umap(df_clusters: pd.DataFrame, output_dir: str):
    """UMAP with Cluster Centroid labels."""
    sns.set(style="white", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_clusters["plot_label"] = df_clusters["case_label"].apply(lambda x: "Control" if str(x).lower() == "control" else "Case")

    # Plot 1: Clusters
    sns.scatterplot(data=df_clusters, x="dim1", y="dim2", hue="cluster", palette="viridis", s=15, alpha=0.5, ax=axes[0], legend=False)
    
    # Annotate Centroids
    for cluster_id in df_clusters['cluster'].unique():
        subset = df_clusters[df_clusters['cluster'] == cluster_id]
        axes[0].text(subset['dim1'].median(), subset['dim2'].median(), str(cluster_id), 
                     fontsize=12, weight='bold', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                     
    axes[0].set_title("A. UMAP Cluster Embeddings", weight='bold')
    axes[0].axis('off')

    # Plot 2: Status
    sns.scatterplot(data=df_clusters, x="dim1", y="dim2", hue="plot_label", palette={"Control": "lightgrey", "Case": "red"}, s=15, alpha=0.6, ax=axes[1])
    axes[1].set_title("B. Case vs Control Distribution", weight='bold')
    axes[1].axis('off')
    axes[1].legend(title="", loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_annotated_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_composition(df_clusters: pd.DataFrame, output_dir: str):
    """100% Stacked Bar Chart for Cluster Purity."""
    sns.set(style="whitegrid", context="paper")
    
    df_clusters["plot_label"] = df_clusters["case_label"].apply(lambda x: "Control" if str(x).lower() == "control" else "Case")
    composition = df_clusters.groupby('cluster')['plot_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    ax = composition[['Case', 'Control']].plot(kind='bar', stacked=True, color=['#e74c3c', '#bdc3c7'], figsize=(8, 5))
    plt.title("Cluster Case/Control Composition", weight='bold')
    plt.ylabel("Percentage (%)")
    plt.xlabel("Cluster ID")
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    sns.despine(left=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_cluster_composition.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lab_zscore_heatmap(heatmap_lab: pd.DataFrame, output_dir: str):
    """Row-standardized heatmap of median lab values."""
    if heatmap_lab is None or heatmap_lab.empty: return
    if -1 in heatmap_lab.columns:
        heatmap_lab = heatmap_lab.drop(columns=[-1])

    heatmap_lab = heatmap_lab.dropna(how='all')
    if heatmap_lab.empty: return

    # Rename columns to "Cluster N" for readability
    heatmap_lab = heatmap_lab.rename(columns={c: f"Cluster {c}" for c in heatmap_lab.columns})

    # Z-Score scale by row (Lab type)
    scaler = StandardScaler()
    z_scored_data = scaler.fit_transform(heatmap_lab.T).T
    z_scored_df = pd.DataFrame(z_scored_data, index=heatmap_lab.index, columns=heatmap_lab.columns)

    plt.figure(figsize=(8, max(6, len(z_scored_df) * 0.3)))
    sns.heatmap(z_scored_df, cmap="coolwarm", center=0, annot=heatmap_lab, fmt=".1f",
                cbar_kws={'label': 'Row Z-Score (Std Dev from Mean)'}, annot_kws={"size": 9})
    plt.title("Lab Value Signatures (Median)", weight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_lab_zscore_heatmap.png"), dpi=300)
    plt.close()

def plot_clinical_heatmap(heatmap_cat: pd.DataFrame, output_dir: str):
    """Heatmap of Categorical Enrichment Scores."""
    if heatmap_cat is None or heatmap_cat.empty: return
    if -1 in heatmap_cat.columns:
        heatmap_cat = heatmap_cat.drop(columns=[-1])

    # Filter to only show highly enriched terms to save space in the paper
    top_heatmap = heatmap_cat[heatmap_cat.max(axis=1) >= 1.5]
    if top_heatmap.empty: return

    # Rename columns to "Cluster N" for readability
    top_heatmap = top_heatmap.rename(columns={c: f"Cluster {c}" for c in top_heatmap.columns})

    plt.figure(figsize=(8, max(8, len(top_heatmap) * 0.25)))
    sns.heatmap(top_heatmap, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'Enrichment Score (x-fold)'})
    plt.title("Clinical Phenotypes (Enrichment > 1.5x)", weight='bold')
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_clinical_fingerprint_heatmap.png"), dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("1. Loading dataset and embeddings...")
    dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(CONFIG_PATH, "held_out")
    embeddings, patient_ids = load_embeddings(EMBEDDINGS_DIR)

    print("2. Running Fixed HDBSCAN Clustering...")
    cluster_labels, z_2d, reducer_10d, reducer_2d = run_fixed_clustering(embeddings)

    joblib.dump(reducer_10d, os.path.join(OUTPUT_DIR, "umap_10d.joblib"))
    joblib.dump(reducer_2d, os.path.join(OUTPUT_DIR, "umap_2d.joblib"))

    df_clusters = pd.DataFrame({"patient_id": patient_ids, "cluster": cluster_labels, "dim1": z_2d[:, 0], "dim2": z_2d[:, 1]})
    df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
    df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
    df_clusters["case_label"] = df_clusters["label_id"].map(label_to_name).fillna("Control")

    print("3. Analyzing Differential Features...")
    df_summary = analyse_differential_features(dataset, df_clusters, subject_to_record_idx, OUTPUT_DIR)

    print("4. Generating Journal Tables (Categorical vs Continuous)...")
    heatmap_cat, heatmap_lab = generate_journal_reports(df_summary, OUTPUT_DIR)

    print("5. Generating High-Res Journal Plots...")
    plot_annotated_umap(df_clusters, OUTPUT_DIR)
    plot_stacked_composition(df_clusters, OUTPUT_DIR)
    plot_clinical_heatmap(heatmap_cat, OUTPUT_DIR)
    plot_lab_zscore_heatmap(heatmap_lab, OUTPUT_DIR)

    print(f"Pipeline Complete! Check {OUTPUT_DIR} for publication-ready figures.")

if __name__ == "__main__":
    main()