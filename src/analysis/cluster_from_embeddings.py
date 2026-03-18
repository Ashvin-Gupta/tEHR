"""
Cluster analysis starting from saved embeddings (no GPU required).

Pipeline:
    - Load embeddings.npy and patient_ids.npy from a clustering run.
    - Run UMAP on original embedding space for visualization only.
    - Run k-means on the original embedding space.
    - Visualize clusters in 2D.
    - Analyse events and measurements per cluster, including continuous lab values.
"""

import os
import yaml
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import hdbscan

from src.data.dataset_factory import get_dataset


# --- CONFIG: update these paths for your run ---
CONFIG_PATH = "src/configs/classification_config.yaml"   # same config you used for training
OUTPUT_DIR = "/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-1-month-logistic-refactored-new-time-mistral-7B/patient_clustering_heldout"  # directory containing embeddings.npy
SPLIT = "held_out"  # "train", "tuning", or "held_out"
K = 5               # number of clusters for k-means
my_terms = ["Secondary polycythaemia", "ADHD", "Benign prostatic hyperplasia", "Bipolar affective disorder", "Bisphosphonates", "Black", "COPD", "Cervical intraepithelial neoplasia", "Chronic cystitis", "Chronic hepatitis", "Chronic kidney disease", "Cleidocranial dysplasia", "Coronary heart disease", "Covid2", "Diabetes", "Drinker - excess/disorder", "Drinker - unspecified", 
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

def load_embeddings(output_dir: str):
    emb_path = os.path.join(output_dir, "embeddings.npy")
    embeddings = np.load(emb_path)

    patient_ids_path = os.path.join(output_dir, "patient_ids.npy")
    if os.path.exists(patient_ids_path):
        patient_ids = np.load(patient_ids_path, allow_pickle=True).tolist()
    else:
        patient_ids = list(range(embeddings.shape[0]))

    return embeddings, patient_ids


def run_umap(embeddings: np.ndarray):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        # random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_kmeans(embeddings: np.ndarray, k: int):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    print(f"Silhouette score (original space, k={k}): {sil:.3f}")
    return labels


def load_dataset(config_path: str, split: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    dataset_name = data_config.get("dataset", "CPRD")

    dataset = get_dataset(dataset_name, data_config, split)

    # Map subject_id -> label_id and label_id -> human-readable label
    subject_to_label = {str(k): v for k, v in dataset.subject_to_label.items()}
    label_to_name = {v: k for k, v in dataset.label_to_id.items()}

    # Map subject_id -> record index (string keys to be consistent)
    subject_to_record_idx = {}
    for idx, record in enumerate(dataset.patient_records):
        sid = record.get("subject_id")
        if sid is not None:
            subject_to_record_idx[str(sid)] = idx

    return dataset, subject_to_label, label_to_name, subject_to_record_idx


def visualize_clusters(df_clusters: pd.DataFrame, output_dir: str):
    sns.set(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Prepare a binary column for plotting to avoid Palette KeyErrors
    # This keeps the original 'case_label' intact for analysis but simplifies for the plot
    df_clusters['plot_label'] = df_clusters['case_label'].apply(
        lambda x: 'Control' if str(x).lower() == 'control' else 'Case'
    )

    # Left: UMAP colored by cluster
    # Using 'viridis' or 'tab20' handles large numbers of clusters better than 'tab10'
    sns.scatterplot(
        data=df_clusters,
        x="dim1",
        y="dim2",
        hue="cluster",
        palette="viridis", 
        s=10,
        alpha=0.6,
        ax=axes[0],
        legend="full"
    )
    axes[0].set_title("UMAP: Clusters")
    axes[0].legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize='small', ncol=2 if len(df_clusters['cluster'].unique()) > 20 else 1)

    # Right: UMAP colored by Case vs Control
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
    plt.savefig(os.path.join(output_dir, "umap_clusters_visualization.png"), dpi=300)
    plt.close()


def analyse_events_and_measurements(
    dataset,
    df_clusters: pd.DataFrame,
    subject_to_record_idx: dict,
    output_dir: str,
):
    """
    Analyse which events and measurements are more frequent / have different values per cluster.

    For measurement-like events (LAB//, MEASUREMENT//, MEDICAL//BMI, MEDICAL//bp_),
    we:
        - Count how often the test is done.
        - Parse the following numeric token as the value when possible (e.g. '91.5').
        - Compute simple summary stats (mean, median, quartiles) across clusters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # cluster -> event_key -> count
    cluster_event_counts = defaultdict(lambda: defaultdict(int))
    # cluster -> event_key -> list of numeric values
    cluster_event_values = defaultdict(lambda: defaultdict(list))

    for _, row in df_clusters.iterrows():
        pid = row["patient_id"]
        cluster = int(row["cluster"])
        pid_key = str(pid)

        rec_idx = subject_to_record_idx.get(pid_key)
        if rec_idx is None:
            continue

        record = dataset.patient_records[rec_idx]
        token_ids = get_filtered_token_ids(dataset, record)
        string_codes = [str(dataset.id_to_token.get(tid, "")) for tid in token_ids]

        i = 0
        while i < len(string_codes):
            code = string_codes[i]

            is_measurement = (
                code.startswith("LAB//")
                or code.startswith("MEASUREMENT//")
                or code.startswith("MEDICAL//BMI")
                or code.startswith("MEDICAL//bp_")
            )

            event_key = None
            if is_measurement:
                event_key = code  # keep raw code; you can later map to human-readable

            # Generic medical events where frequency alone may be interesting
            elif code.startswith("MEDICAL//"):
                event_key = code

            if event_key is not None:
                cluster_event_counts[cluster][event_key] += 1

                # Try to parse the next token as a numeric value (continuous lab)
                if is_measurement and i + 1 < len(string_codes):
                    next_code = string_codes[i + 1]
                    try:
                        value = float(next_code)
                        cluster_event_values[cluster][event_key].append(value)
                    except ValueError:
                        # Non-numeric (e.g. a bin like 'low/normal/high'); skip for numeric stats
                        pass

            i += 1

    # Build per-cluster CSVs and a global summary
    summary_rows = []
    for cluster_id, event_counts in cluster_event_counts.items():
        rows = []
        for event_key, count in event_counts.items():
            values = cluster_event_values[cluster_id].get(event_key, [])
            n_values = len(values)

            if n_values > 0:
                arr = np.array(values, dtype=float)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                median = float(np.median(arr))
                q25 = float(np.percentile(arr, 25))
                q75 = float(np.percentile(arr, 75))
            else:
                mean = std = median = q25 = q75 = np.nan

            # Optionally, translate event_key to something human readable
            human_readable = event_key
            try:
                human_readable = dataset.translator.translate(event_key).rstrip("; ")
            except Exception:
                pass

            row = {
                "cluster": cluster_id,
                "event_code": event_key,
                "event_human": human_readable,
                "count": count,
                "n_values": n_values,
                "mean_value": mean,
                "std_value": std,
                "median_value": median,
                "q25_value": q25,
                "q75_value": q75,
            }
            rows.append(row)
            summary_rows.append(row)

        df_cluster = pd.DataFrame(rows)
        df_cluster.sort_values("count", ascending=False, inplace=True)
        df_cluster.to_csv(
            os.path.join(output_dir, f"cluster_{cluster_id}_events_measurements.csv"),
            index=False,
        )

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(
            os.path.join(output_dir, "cluster_events_measurements_summary.csv"),
            index=False,
        )

def calculate_cluster_composition(df_clusters: pd.DataFrame, output_dir: str):
    # CRITICAL FIX: Any label that is NOT 'Control' (0) is a Case.
    df_clusters['binary_label'] = df_clusters['case_label'].apply(
        lambda x: 'Control' if str(x).lower() == 'control' else 'Case'
    )
    
    # Create the pivot table
    composition = df_clusters.groupby('cluster')['binary_label'].value_counts().unstack(fill_value=0)
    
    # Ensure columns exist even if one cluster has 0 cases or 0 controls
    for col in ['Case', 'Control']:
        if col not in composition.columns:
            composition[col] = 0
            
    composition['Total'] = composition['Case'] + composition['Control']
    composition['Case_%'] = (composition['Case'] / composition['Total'] * 100).round(2)
    
    print("\n--- Optimized Cluster Composition ---")
    print(composition.sort_values('Total', ascending=False))
    
    composition.to_csv(os.path.join(output_dir, "cluster_case_control_composition.csv"))
    return composition

def analyse_differential_features(
    dataset,
    df_clusters: pd.DataFrame,
    subject_to_record_idx: dict,
    output_dir: str,
):
    """
    Analyzes which events/measurements are unique or statistically different 
    in each cluster compared to the rest of the population.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect all data into a flat structure for global comparison
    all_data = []
    
    for _, row in df_clusters.iterrows():
        pid = str(row["patient_id"])
        cluster = int(row["cluster"])
        rec_idx = subject_to_record_idx.get(pid)
        if rec_idx is None: continue

        record = dataset.patient_records[rec_idx]
        token_ids = get_filtered_token_ids(dataset, record)
        tokens = [str(dataset.id_to_token.get(tid, "")) for tid in token_ids]

        for i, code in enumerate(tokens):
            is_measurement = any(code.startswith(prefix) for prefix in ["LAB//", "MEASUREMENT//", "MEDICAL//BMI", "MEDICAL//bp_"])
            is_medical = code.startswith("MEDICAL//")

            if is_measurement or is_medical:
                val = np.nan
                if is_measurement and i + 1 < len(tokens):
                    try: val = float(tokens[i+1])
                    except ValueError: pass
                
                all_data.append({
                    'cluster': cluster,
                    'event_code': code,
                    'value': val
                })

    df_raw = pd.DataFrame(all_data)
    
    # 2. Compute Global Stats (for comparison)
    global_counts = df_raw['event_code'].value_counts()
    global_medians = df_raw.groupby('event_code')['value'].median()
    total_patients = df_clusters['patient_id'].nunique()

    # 3. Compute Per-Cluster Stats and compare to Global
    summary_rows = []
    
    for cluster_id in sorted(df_clusters['cluster'].unique()):
        cluster_df = df_raw[df_raw['cluster'] == cluster_id]
        cluster_size = df_clusters[df_clusters['cluster'] == cluster_id]['patient_id'].nunique()
        
        counts = cluster_df['event_code'].value_counts()
        
        for event_code, count in counts.items():
            vals = cluster_df[cluster_df['event_code'] == event_code]['value'].dropna()
            
            # Frequency Analysis
            # How much more common is this event here than in the general population?
            global_freq = global_counts[event_code] / total_patients
            cluster_freq = count / cluster_size
            enrichment_factor = cluster_freq / global_freq if global_freq > 0 else 0

            # Value Analysis
            mean_v = vals.mean() if not vals.empty else np.nan
            median_v = vals.median() if not vals.empty else np.nan
            
            # Delta from global median (instead of mean)
            diff_from_global_median = median_v - global_medians[event_code] if not np.isnan(median_v) else 0

            try:
                human = dataset.translator.translate(event_code).rstrip("; ")
            except Exception:
                human = event_code

            summary_rows.append({
                "cluster": cluster_id,
                "event_code": event_code,
                "event_human": human,
                "count": count,
                "enrichment_score": round(enrichment_factor, 3), # >1 means more common in this cluster
                "mean_value": mean_v,
                "median_value": median_v,
                "diff_from_global_median": diff_from_global_median,
                "q25_value": vals.quantile(0.25) if not vals.empty else np.nan,
                "q75_value": vals.quantile(0.75) if not vals.empty else np.nan,
            })

    df_final = pd.DataFrame(summary_rows)
    
    # Sort by cluster and then by enrichment to see "Top characteristic features"
    df_final = df_final.sort_values(['cluster', 'enrichment_score'], ascending=[True, False])
    
    # Save the full summary
    df_final.to_csv(os.path.join(output_dir, "differential_cluster_analysis.csv"), index=False)
    
    # Save individual files for each cluster for easy reading
    for cid in df_clusters['cluster'].unique():
        cluster_subset = df_final[df_final['cluster'] == cid]
        cluster_subset.to_csv(os.path.join(output_dir, f"cluster_{cid}_analysis.csv"), index=False)

    print(f"Differential analysis complete. Files saved to {output_dir}")

def get_filtered_token_ids(dataset, record: dict) -> list:
    """Return token IDs for a record after applying time cutoff and removing cancer tokens."""
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


def run_clustering_pipeline(embeddings: np.ndarray, min_cluster_size=50):
    print(f"Reducing {embeddings.shape[1]}D to 10D...")
    reducer_10d = umap.UMAP(n_components=10, n_neighbors=50, min_dist=0.0, random_state=42)
    embeddings_10d = reducer_10d.fit_transform(embeddings)

    print("Running HDBSCAN (Optimizing for stable clusters)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, # Increased to merge micro-clusters
        min_samples=5,                    # Increased to filter out more noise
        cluster_selection_method='leaf',    # 'eom' finds larger, more stable groups, leaf finds smaller, more specific clusters
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings_10d)
    
    # 2D for the plot
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    z_2d = reducer_2d.fit_transform(embeddings)
    
    return labels, z_2d

def generate_fingerprint_and_signatures(df_summary, terms_list, output_dir):
    """Generates the Heatmap and the Top-10 Global Signature report."""
    # 1. Top 10 Global Signatures per cluster
    global_top_10 = []
    for cid in sorted(df_summary['cluster'].unique()):
        # Filter for codes that appear enough to be relevant
        top = df_summary[df_summary['cluster'] == cid].sort_values('enrichment_score', ascending=False).head(10)
        global_top_10.append(top)
    
    pd.concat(global_top_10).to_csv(os.path.join(output_dir, "top_10_global_signatures.csv"), index=False)

    # 2. Terms of Interest Heatmap
    rows = []
    for term in list(dict.fromkeys(terms_list)):
        match = df_summary[df_summary['event_human'] == term].copy()
        if match.empty: continue
        # Take most frequent code if duplicates exist
        match = match.sort_values('count', ascending=False).drop_duplicates(subset=['cluster'])
        rows.append(match)

    if rows:
        df_filtered = pd.concat(rows)
        # Numeric matrix for heatmap
        heatmap_data = df_filtered.pivot(index='event_human', columns='cluster', values='enrichment_score').fillna(0)
        
        # String matrix for the CSV pivot
        df_filtered['display'] = df_filtered.apply(
            lambda r: f"{r['median_value']:.1f}" if pd.notnull(r['median_value']) and r['median_value'] != 0 
            else f"x{r['enrichment_score']:.1f}", axis=1
        )
        pivot = df_filtered.pivot(index='event_human', columns='cluster', values='display').fillna("-")
        pivot.to_csv(os.path.join(output_dir, "clinical_fingerprint_pivot.csv"))
        
        return heatmap_data
    return None

def generate_clinical_reports(df_summary, terms_list, output_dir):
    """
    1. Produces a Global Top-20 enrichment report per cluster.
    2. Produces the specific Term-of-Interest pivot table.
    """
    # --- Part 1: Global Signatures ---
    global_report = []
    for cid in sorted(df_summary['cluster'].unique()):
        # Get top 20 most enriched codes that appear at least 5 times
        top_codes = df_summary[
            (df_summary['cluster'] == cid) & 
            (df_summary['count'] > 5)
        ].sort_values('enrichment_score', ascending=False).head(20)
        global_report.append(top_codes)
    
    pd.concat(global_report).to_csv(os.path.join(output_dir, "global_cluster_signatures_hdbscan.csv"), index=False)

    # --- Part 2: Terms of Interest Matrix ---
    rows = []
    for term in list(dict.fromkeys(terms_list)):
        match = df_summary[df_summary['event_human'] == term].copy()
        if match.empty: continue
        
        # Deduplicate: if "BMI" appears twice in cluster 1, take the one with higher count
        match = match.sort_values('count', ascending=False).drop_duplicates(subset=['cluster'])
        rows.append(match)

    if rows:
        df_filtered = pd.concat(rows)
        # Create display string
        df_filtered['display'] = df_filtered.apply(
            lambda r: f"{r['median_value']:.1f}" if pd.notnull(r['median_value']) and r['median_value'] != 0 
            else f"x{r['enrichment_score']:.1f}", axis=1
        )
        
        pivot = df_filtered.pivot(index='event_human', columns='cluster', values='display').fillna("-")
        pivot.to_csv(os.path.join(output_dir, "terms_of_interest_pivot_hdbscan.csv"))
        
        # Numeric heatmap data
        heatmap_data = df_filtered.pivot(index='event_human', columns='cluster', values='enrichment_score').fillna(0)
        return pivot, heatmap_data
    return None, None

def plot_clinical_heatmap(heatmap_data, output_dir):
    if heatmap_data is None or heatmap_data.empty: return
    
    # Remove the 'Noise' cluster from the heatmap for better contrast
    if -1 in heatmap_data.columns:
        heatmap_data = heatmap_data.drop(columns=[-1])

    plt.figure(figsize=(14, max(10, len(heatmap_data) * 0.25)))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Enrichment Score'})
    plt.title("Clinical Fingerprint: Enrichment by Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clinical_fingerprint_heatmap.png"), dpi=300)
# def main():
#     # 1) Load embeddings and run clustering
#     embeddings, patient_ids = load_embeddings(OUTPUT_DIR)
#     Z = run_umap(embeddings)
#     cluster_labels = run_kmeans(embeddings, K)

#     # 2) Build cluster DataFrame
#     df_clusters = pd.DataFrame(
#         {
#             "patient_id": patient_ids,
#             "cluster": cluster_labels,
#             "dim1": Z[:, 0],
#             "dim2": Z[:, 1],
#         }
#     )

#     # 3) Load dataset to attach labels and access raw tokens
#     dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(
#         CONFIG_PATH, SPLIT
#     )

#     # Map labels
#     df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
#     df_clusters["label_id"] = df_clusters["patient_id_str"].map(subject_to_label)
#     df_clusters["label_name"] = df_clusters["label_id"].map(label_to_name)

#     # Binary case vs control
#     df_clusters["is_case"] = df_clusters["label_id"].fillna(0).astype(int) > 0
#     df_clusters["case_label"] = np.where(df_clusters["is_case"], "Case", "Control")

#     # Save clusters table
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     df_clusters.to_csv(
#         os.path.join(OUTPUT_DIR, "patient_clusters_from_embeddings.csv"), index=False
#     )

#     # # 4) Visualise clusters
#     # visualize_clusters(df_clusters, OUTPUT_DIR)

#     # # 5) Analyse events and measurements per cluster
#     # analyse_events_and_measurements(
#     #     dataset=dataset,
#     #     df_clusters=df_clusters,
#     #     subject_to_record_idx=subject_to_record_idx,
#     #     output_dir=OUTPUT_DIR,
#     # )
#     # 4) Calculate Case/Control mix
#     calculate_cluster_composition(df_clusters, OUTPUT_DIR)

#     # 5) Run the new differential analysis
#     analyse_differential_features(
#         dataset=dataset,
#         df_clusters=df_clusters,
#         subject_to_record_idx=subject_to_record_idx,
#         output_dir=OUTPUT_DIR
#     )

#     # 6) Visualise (as before)
#     visualize_clusters(df_clusters, OUTPUT_DIR)

#     print("Cluster analysis from embeddings complete.")

def main():
    # 1) Load data & dataset
    dataset, subject_to_label, label_to_name, subject_to_record_idx = load_dataset(CONFIG_PATH, SPLIT)
    embeddings, patient_ids = load_embeddings(OUTPUT_DIR)
    
    # 2) Stricter clustering (min_cluster_size=15 usually gives more granularity)
    cluster_labels, Z = run_clustering_pipeline(embeddings, min_cluster_size=15)
    
    # 3) Build DataFrame
    df_clusters = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster": cluster_labels,
        "dim1": Z[:, 0],
        "dim2": Z[:, 1],
    })
    df_clusters["patient_id_str"] = df_clusters["patient_id"].astype(str)
    df_clusters["case_label"] = df_clusters["patient_id_str"].map(subject_to_label).map(label_to_name).fillna("Control")

    # 4) Basic Analysis
    calculate_cluster_composition(df_clusters, OUTPUT_DIR)
    analyse_differential_features(dataset, df_clusters, subject_to_record_idx, OUTPUT_DIR)
    
    # 5) Specific Reports & Heatmap
    df_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "differential_cluster_analysis.csv"))
    heatmap_df = generate_fingerprint_and_signatures(df_summary, my_terms, OUTPUT_DIR)

    # 6) Visualization
    visualize_clusters(df_clusters, OUTPUT_DIR)
    plot_clinical_heatmap(heatmap_df, OUTPUT_DIR)

    print(f"Pipeline complete. Check {OUTPUT_DIR} for analysis and images.")

if __name__ == "__main__":
    main()