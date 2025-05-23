#ADDECODE
    
    # Preprocess all risk data the same way as we did with healthy on previous script
    # Zscore only using healthy data !!


    #SHAP EDGES


        #BEESWARM colored by ages and risk groups
        #BEESWARM separated by ages and risk group



#################  IMPORT NECESSARY LIBRARIES  ################


import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions

import torch
import random
import numpy as np

import networkx as nx  # For graph-level metrics

# === Set seed for reproducibility ===
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)



# ADDECODE Data

####################### CONNECTOMES ###############################
print("ADDECODE CONNECTOMES\n")

# === Define paths ===
zip_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_connectome_act.zip"
directory_inside_zip = "connectome_act/"
connectomes = {}

# === Load connectome matrices from ZIP ===
with zipfile.ZipFile(zip_path, 'r') as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                df = pd.read_csv(f, header=None)
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[subject_id] = df

print(f"Total connectome matrices loaded: {len(connectomes)}")

# === Filter out connectomes with white matter on their file name ===
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# === Extract subject IDs from filenames ===
cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)
    if match:
        num_id = match.group(1).zfill(5)  # Ensure 5-digit IDs
        cleaned_connectomes[num_id] = v

print()



############################## METADATA ##############################


print("ADDECODE METADATA\n")

# === Load metadata CSV ===
metadata_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx"
df_metadata = pd.read_excel(metadata_path)

# === Generate standardized subject IDs → 'DWI_fixed' (e.g., 123 → '00123')
df_metadata["MRI_Exam_fixed"] = (
    df_metadata["MRI_Exam"]
    .fillna(0)                           # Handle NaNs first
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# === Drop fully empty rows and those with missing DWI ===
df_metadata_cleaned = df_metadata.dropna(how='all')                       # Remove fully empty rows
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["MRI_Exam"])         # Remove rows without DWI

# === Display result ===
print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()





#################### MATCH CONNECTOMES & METADATA ####################

print(" MATCHING CONNECTOMES WITH METADATA")

# === Filter metadata to only subjects with connectomes available ===
matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

# === Build dictionary of matched connectomes ===
matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}


# === Store matched metadata as a DataFrame for further processing ===
df_matched_connectomes = matched_metadata.copy()





# df_matched_connectomes:
# → Cleaned metadata that has a valid connectome
# → Includes AD/MCI

# matched_connectomes:
# → Dictionary of connectomes that have valid metadata
# → Key: subject ID
# → Value: connectome matrix
# → Includes AD/MCI




# df_matched_addecode_healthy:
# → Metadata of only healthy subjects (no AD/MCI)
# → Subset of df_matched_connectomes

# matched_connectomes_healthy_addecode:
# → Connectomes of only healthy subjects
# → Subset of matched_connectomes





########### PCA GENES ##########

print("PCA GENES")

import pandas as pd

# Read 
df_pca = pd.read_csv("/home/bas/Desktop/MyData/AD_DECODE/PCA_genes/PCA_human_blood_top30.csv")
print(df_pca.head())

print(df_matched_connectomes.head())



# Fix id formats

# === Fix ID format in PCA DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE_1' → 'ADDECODE1'
df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)



# === Fix Subject format in metadata DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE1' → 'ADDECODE1'
df_matched_connectomes["IDRNA_fixed"] = df_matched_connectomes["IDRNA"].str.upper().str.replace("_", "", regex=False)




###### MATCH PCA GENES WITH METADATA############

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_withConnectome = df_matched_connectomes.merge(df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed")


#Numbers

# === Show how many subjects with PCA and connectome you have
print(f" subjects with metadata connectome: {df_matched_connectomes.shape[0]}")
print()

print(f" subjects with metadata PCA & connectome: {df_metadata_PCA_withConnectome.shape[0]}")
print()


# Get the full set of subject IDs (DWI_fixed) in set
all_ids = set(df_matched_connectomes["MRI_Exam_fixed"])

# Get the subject IDs (DWI_fixed) that matched with PCA
with_pca_ids = set(df_metadata_PCA_withConnectome["MRI_Exam_fixed"])

# Compute the difference:  subjects without PCA
without_pca_ids = all_ids - with_pca_ids

# Filter the original metadata for those subjects
df_without_pca = df_matched_connectomes[
    df_matched_connectomes["MRI_Exam_fixed"].isin(without_pca_ids)
]


# Print result
print(f" Subjects with connectome but NO PCA: {df_without_pca.shape[0]}")
print()

print(df_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]])






####################### FA MD Vol #############################



# === Load FA data ===
fa_path = "/home/bas/Desktop/MyData/AD_DECODE/RegionalStats/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_fa.txt"
df_fa = pd.read_csv(fa_path, sep="\t")
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"]
df_fa = df_fa.reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col.startswith("S")]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)

# === Load MD data ===
md_path = "/home/bas/Desktop/MyData/AD_DECODE/RegionalStats/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_md.txt"
df_md = pd.read_csv(md_path, sep="\t")
df_md = df_md[1:]
df_md = df_md[df_md["ROI"] != "0"]
df_md = df_md.reset_index(drop=True)
subject_cols_md = [col for col in df_md.columns if col.startswith("S")]
df_md_transposed = df_md[subject_cols_md].transpose()
df_md_transposed.columns = [f"ROI_{i+1}" for i in range(df_md_transposed.shape[1])]
df_md_transposed.index.name = "subject_id"
df_md_transposed = df_md_transposed.astype(float)

# === Load Volume data ===
vol_path = "/home/bas/Desktop/MyData/AD_DECODE/RegionalStats/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"]
df_vol = df_vol.reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col.startswith("S")]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)


# === Combine FA + MD + Vol per subject ===
multimodal_features_dict = {}

for subj in df_fa_transposed.index:
    subj_id = subj.replace("S", "").zfill(5)
    if subj in df_md_transposed.index and subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float)
        stacked = torch.stack([fa, md, vol], dim=1)  # Shape: [84, 3]
        multimodal_features_dict[subj_id] = stacked

# === Normalization nodo-wise between subjects ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # [N_subjects, 84, 3]
    means = all_features.mean(dim=0)  # [84, 3]
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Apply normalization

# === Get IDs of healthy subjects only (exclude AD and MCI)
healthy_ids = df_matched_connectomes[~df_matched_connectomes["Risk"].isin(["AD", "MCI"])]["MRI_Exam_fixed"].tolist()

# === Stack node features only from healthy subjects
healthy_stack = torch.stack([
    multimodal_features_dict[subj]
    for subj in healthy_ids
    if subj in multimodal_features_dict
])  # Shape: [N_healthy, 84, 3]

# === Compute mean and std from healthy controls
node_means = healthy_stack.mean(dim=0)  # [84, 3]
node_stds = healthy_stack.std(dim=0) + 1e-8

# === Apply normalization to ALL subjects using healthy stats
normalized_node_features_dict = {
    subj: (features - node_means) / node_stds
    for subj, features in multimodal_features_dict.items()
}






# === Function to compute clustering coefficient per node ===
def compute_nodewise_clustering_coefficients(matrix):
    """
    Compute clustering coefficient for each node in the connectome matrix.
    
    Parameters:
        matrix (pd.DataFrame): 84x84 connectivity matrix
    
    Returns:
        torch.Tensor: Tensor of shape [84, 1] with clustering coefficient per node
    """
    G = nx.from_numpy_array(matrix.to_numpy())

    # Assign weights from matrix to the graph
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]

    # Compute clustering coefficient per node
    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]

    # Convert to tensor [84, 1]
    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)








# ===============================
# Step 9: Threshold and Log Transform Connectomes
# ===============================

import numpy as np
import pandas as pd

# --- Define thresholding function ---
def threshold_connectome(matrix, percentile=100):
    """
    Apply percentile-based thresholding to a connectome matrix.
    """
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

# --- Apply threshold + log transform ---
log_thresholded_connectomes = {}
for subject, matrix in matched_connectomes.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)



##################### MATRIX TO GRAPH FUNCTION #######################

import torch
import numpy as np
from torch_geometric.data import Data


# === Function to convert a connectome matrix into a graph with multimodal node features ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)

    # === Get FA, MD, Volume features [84, 3]
    node_feats = node_features_dict[subject_id]

    # === Compute clustering coefficient per node [84, 1]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)

    # === Concatenate and scale [84, 4]
    full_node_features = torch.cat([node_feats, clustering_tensor], dim=1)
    node_features = 0.5 * full_node_features.to(device)

    return edge_index, edge_attr, node_features






# ===============================
# Step 10: Compute Graph Metrics and Add to Metadata
# ===============================

import networkx as nx

# --- Define graph metric functions ---
def compute_clustering_coefficient(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.average_clustering(G, weight="weight")

def compute_path_length(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        weight = matrix.iloc[u, v]
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        return nx.average_shortest_path_length(G, weight="distance")
    except:
        return float("nan")

# --- Assign computed metrics to metadata ---
addecode_metadata_pca = df_metadata_PCA_withConnectome.reset_index(drop=True)
addecode_metadata_pca["Clustering_Coeff"] = np.nan
addecode_metadata_pca["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "Clustering_Coeff"
        ] = clustering
        addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "Path_Length"
        ] = path
    except Exception as e:
        print(f"Failed to compute metrics for subject {subject}: {e}")


# ===============================
# Step 11: Normalize Metadata and PCA Columns
# ===============================

from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

#label encoding sex
le_sex = LabelEncoder()
addecode_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_metadata_pca["sex"].astype(str))


# --- Label encode genotype ---
le = LabelEncoder()
addecode_metadata_pca["genotype"] = le.fit_transform(addecode_metadata_pca["genotype"].astype(str))

# --- Normalize numerical and PCA columns ---
numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3'] #Top 10 from SPEARMAN  corr (enrich)


# === Select healthy subjects only
df_controls = addecode_metadata_pca[~addecode_metadata_pca["Risk"].isin(["AD", "MCI"])]

# === Define all columns to z-score
all_zscore_cols = numerical_cols + pca_cols

# === Compute means and stds using only healthy controls
global_means = df_controls[all_zscore_cols].mean()
global_stds = df_controls[all_zscore_cols].std() + 1e-8

# === Apply normalization to ALL subjects using healthy stats
addecode_metadata_pca[all_zscore_cols] = (
    addecode_metadata_pca[all_zscore_cols] - global_means
) / global_stds




# ===============================
# Step 12: Build Metadata, graph metrics and PCA Tensors
# ===============================

# === 1. Demographic tensor (systolic, diastolic, sex one-hot, genotype) ===
subject_to_demographic_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Systolic"],
        row["Diastolic"],
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in addecode_metadata_pca.iterrows()
}

# === 2. Graph metric tensor (clustering coefficient, path length) ===
subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_metadata_pca.iterrows()
}

# === 3. PCA tensor (top 10 age-correlated components) ===
subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_metadata_pca.iterrows()
}




#################  CONVERT MATRIX TO GRAPH  ################

graph_data_list_addecode = []
final_subjects_with_all_data = []  #verify subjects

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        # === Skip if any required input is missing ===
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_graphmetric_tensor:
            continue
        if subject not in subject_to_pca_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # === Convert matrix to graph (node features: FA, MD, Vol, clustering)
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device=torch.device("cpu"), subject_id=subject, node_features_dict=normalized_node_features_dict
        )

        # === Get target age
        age_row = addecode_metadata_pca.loc[
            addecode_metadata_pca["MRI_Exam_fixed"] == subject, "age"
        ]
        if age_row.empty:
            continue
        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        # === Concatenate global features (demographics + graph metrics + PCA)
        demo_tensor = subject_to_demographic_tensor[subject]     # [5]
        graph_tensor = subject_to_graphmetric_tensor[subject]    # [2]
        pca_tensor = subject_to_pca_tensor[subject]              # [10]
        
        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)  # [16]

        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0)  # Shape: (1, 16)
        )
        data.subject_id = subject  # Track subject

        # === Store graph
        graph_data_list_addecode.append(data)
        final_subjects_with_all_data.append(subject)
        
        # === Print one example to verify shapes and content
        if len(graph_data_list_addecode) == 1:
            print("\n Example PyTorch Geometric Data object:")
            print("→ Node features shape:", data.x.shape)           # Expected: [84, 4]
            print("→ Edge index shape:", data.edge_index.shape)     # Expected: [2, ~3500]
            print("→ Edge attr shape:", data.edge_attr.shape)       # Expected: [~3500]
            print("→ Global features shape:", data.global_features.shape)  # Ecpected: [1, 16]
            print("→ Target age (y):", data.y.item())


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")






# 1) AGE

# 1.1 TOP 10 beeswarm colored by age group (young middle old)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path where the SHAP CSVs are stored
shap_dir = "/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/4.1_ SHAP edges"

# Load all SHAP CSVs and attach subject + age info
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        subj_id = fname.split("_")[-1].replace(".csv", "")
        df["subject"] = subj_id

        # Attach age and group from metadata
        age_row = addecode_metadata_pca[addecode_metadata_pca["MRI_Exam_fixed"] == subj_id]
        if not age_row.empty:
            age = age_row["age"].values[0]
            df["age"] = age

            if age < 45:
                df["group"] = "Young"
            elif age < 65:
                df["group"] = "Middle"
            else:
                df["group"] = "Old"

            shap_dfs.append(df)

# Combine all into one DataFrame
all_df = pd.concat(shap_dfs)





# Region names for edge labels
region_names = [  # ← tu lista de 84 regiones aquí
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

# Safely generate edge labels
def get_edge_label(row):
    try:
        i = int(row["Node_i"])
        j = int(row["Node_j"])
        if 0 <= i < len(region_names) and 0 <= j < len(region_names):
            return f"{region_names[i]} ↔ {region_names[j]}"
        else:
            return f"{i} ↔ {j}"
    except:
        return "Invalid"

all_df["edge_label"] = all_df.apply(get_edge_label, axis=1)

# Top 10 most important connections (mean absolute SHAP)
top_edges = (
    all_df.groupby("edge_label")["SHAP_value"]
    .apply(lambda x: x.abs().mean())
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# Filter for top edges
top_df = all_df[all_df["edge_label"].isin(top_edges)]

# Sort edges for consistent y-axis
edge_order = top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

# Plot beeswarm
plt.figure(figsize=(12, 8))
sns.stripplot(
    data=top_df,
    x="SHAP_value",
    y="edge_label",
    hue="group",
    order=edge_order,
    dodge=True,
    jitter=0.25,
    alpha=0.7
)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("SHAP Value (impact on predicted brain age)")
plt.ylabel("Edge (brain connection)")
plt.title("Top 10 Most SHAP-Important Connections — Colored by Age Group")
plt.legend(title="Group")
plt.tight_layout()
plt.show()









# 1.2 Three beeswarms, by age


import matplotlib.pyplot as plt
import seaborn as sns

# Loop for each age group
for group in ["Young", "Middle", "Old"]:
    group_df = all_df[all_df["group"] == group].copy()

    if group_df.empty:
        print(f" No data for group: {group}")
        continue

    # Compute top 10 edges for this group
    group_top_edges = (
        group_df.groupby("edge_label")["SHAP_value"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    # Filter and sort
    group_top_df = group_df[group_df["edge_label"].isin(group_top_edges)]
    edge_order = group_top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

    print(f" {group}: {len(group_top_df)} rows from top 10 edges")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=group_top_df,
        x="SHAP_value",
        y="edge_label",
        order=edge_order,
        jitter=0.25,
        alpha=0.7,
        color="dodgerblue"
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("SHAP Value (impact on predicted brain age)")
    plt.ylabel("Edge (connection)")
    plt.title(f"Top 10 SHAP-Important Edges — {group} Group")
    plt.tight_layout()
    plt.show()





# 2) RISK GROUP
# 2.1 TOP 10 beeswarm colored by RISK group 


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path where the SHAP CSVs are stored
shap_dir = "/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/4.1_ SHAP edges"  

# Load all SHAP CSVs
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        subj_id = fname.split("_")[-1].replace(".csv", "")
        df["subject"] = subj_id

        # Get age and risk from metadata
        subj_row = addecode_metadata_pca[addecode_metadata_pca["MRI_Exam_fixed"] == subj_id]
        if not subj_row.empty:
            df["age"] = subj_row["age"].values[0]
            df["group"] = subj_row["Risk"].values[0] if not pd.isna(subj_row["Risk"].values[0]) else "NoRisk"
            shap_dfs.append(df)
        else:
            df["age"] = None
            df["group"] = "NoRisk"
            shap_dfs.append(df)  # <- important to append here too!


# Combine all subjects
all_df = pd.concat(shap_dfs)



all_df["edge_label"] = all_df.apply(
    lambda row: f"{region_names[int(row['Node_i'])]} ↔ {region_names[int(row['Node_j'])]}", axis=1
)

# Top 10 edges with highest mean absolute SHAP
top_edges = (
    all_df.groupby("edge_label")["SHAP_value"]
    .apply(lambda x: x.abs().mean())
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# Filter to top edges
top_df = all_df[all_df["edge_label"].isin(top_edges)]

# Order edges by SHAP direction (mean)
edge_order = top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

# Plot beeswarm with risk groups
plt.figure(figsize=(12, 8))
sns.stripplot(
    data=top_df,
    x="SHAP_value",
    y="edge_label",
    hue="group",  # now: Control, Familial, MCI, AD
    order=edge_order,
    dodge=True,
    jitter=0.25,
    alpha=0.7
)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("SHAP Value (contribution to brain age prediction)")
plt.ylabel("Edge (connection)")
plt.title("Top 10 SHAP-Important Edges — Colored by Clinical Group")
plt.legend(title="Risk Group")
plt.tight_layout()
plt.show()







# 2.2 beeswarms by risk group



# Merge Risk group into the SHAP dataframe using subject ID
all_df = all_df.merge(
    addecode_metadata_pca[["MRI_Exam_fixed", "Risk"]],
    left_on="subject",
    right_on="MRI_Exam_fixed",
    how="left"
)

all_df["Risk"] = all_df["Risk"].fillna("NoRisk")



import matplotlib.pyplot as plt
import seaborn as sns

# Loop for each risk group
for risk_group in all_df["Risk"].unique():
    group_df = all_df[all_df["Risk"] == risk_group].copy()

    if group_df.empty:
        print(f" No data for risk group: {risk_group}")
        continue

    # Compute top 10 edges for this risk group
    group_top_edges = (
        group_df.groupby("edge_label")["SHAP_value"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    # Filter and sort
    group_top_df = group_df[group_df["edge_label"].isin(group_top_edges)]
    edge_order = group_top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

    print(f" {risk_group}: {len(group_top_df)} rows from top 10 edges")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=group_top_df,
        x="SHAP_value",
        y="edge_label",
        order=edge_order,
        jitter=0.25,
        alpha=0.7,
        color="indianred"
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("SHAP Value (impact on predicted brain age)")
    plt.ylabel("Edge (connection)")
    plt.title(f"Top 10 SHAP-Important Edges — {risk_group} Group")
    plt.tight_layout()
    plt.show()





