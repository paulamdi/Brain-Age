#SHAP EDGES PLOTS BY AGE AND RISK GROUPS

# LOAD ALL DATA INCLUDING UNHELATHY

#ADRC DTI fmri 2 channels bimodal 
#with rmse
import os
import pandas as pd
import numpy as np
import random
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


#LOAD CONNECTOMES
print ("LOAD CONNECTOMES DTI")

base_path_adrc_dti = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/DTI/plain"
adrc_dti_connectomes = {}

for filename in os.listdir(base_path_adrc_dti):
    # Skip hidden system files or ._macOS files
    if not filename.endswith("_conn_plain.csv") or filename.startswith("._"):
        continue

    subject_id = filename.replace("_conn_plain.csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_dti, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_dti_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC DTI connectomes loaded: {len(adrc_dti_connectomes)}")
print(" Example subject IDs:", list(adrc_dti_connectomes.keys())[:5])
print()





print("LOAD CONNECTOMES fMRI")

base_path_adrc_fmri = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/fMRI/corr"
adrc_fmri_connectomes = {}

for filename in os.listdir(base_path_adrc_fmri):
    if not filename.startswith("func_connectome_corr_ADRC") or not filename.endswith(".csv"):
        continue

    subject_id = filename.replace("func_connectome_corr_", "").replace(".csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_fmri, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_fmri_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC fMRI connectomes loaded: {len(adrc_fmri_connectomes)}")
print(" Example subject IDs:", list(adrc_fmri_connectomes.keys())[:5])
print()



# Intersect subjects that have both DTI and fMRI
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

print(f" Matched subjects with both DTI and fMRI: {len(matched_dti_fmri)}")
print(" Example matched subjects:", matched_dti_fmri[:5])
print()



#LOAD METADATA
print("LOAD METADATA ")


# === Step 1: Define the metadata file path ===
metadata_path_adrc = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/metadata/alex-badea_2024-06-14 (copy).xlsx"

# === Step 2: Load the Excel file into a DataFrame ===
# This will read the first sheet by default
df_adrc_meta= pd.read_excel(metadata_path_adrc)

# === Step 3: Display metadata summary ===
print(" ADRC metadata loaded successfully.")
print(" Metadata shape:", df_adrc_meta.shape)
print(" Preview of first rows:")
print(df_adrc_meta.head())
print()






#MATCH CONNECTMES (has both) AND METADATA
print("=== MATCHING CONNECTOMES AND METADATA (DTI + fMRI only) ===")

# List of subjects that have both DTI and fMRI connectomes
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

# Get subject IDs present in the metadata
metadata_subject_ids = set(df_adrc_meta["PTID"])

# Find subjects that have DTI, fMRI, and metadata
matched_ids = sorted(set(matched_dti_fmri) & metadata_subject_ids)

# Filter metadata to include only matched subjects
df_matched_adrc = df_adrc_meta[df_adrc_meta["PTID"].isin(matched_ids)].copy()

# Filter DTI connectomes to matched subjects
adrc_dti_connectomes_matched = {
    sid: adrc_dti_connectomes[sid]
    for sid in matched_ids
}

# Filter fMRI connectomes to matched subjects
adrc_fmri_connectomes_matched = {
    sid: adrc_fmri_connectomes[sid]
    for sid in matched_ids
}

# Print summary of matching
print(f"Subjects with DTI + fMRI: {len(matched_dti_fmri)}")
print(f"Subjects with metadata: {len(metadata_subject_ids)}")
print(f"Matched subjects (DTI + fMRI + metadata): {len(matched_ids)}")
print(f"- Rows in matched metadata: {df_matched_adrc.shape[0]}")
print(f"- DTI connectomes matched: {len(adrc_dti_connectomes_matched)}")
print(f"- fMRI connectomes matched: {len(adrc_fmri_connectomes_matched)}")
print("Example matched subject IDs:", matched_ids[:5])
print()





#df_matched_adrc
#adrc_dti_connectomes_matched
#adrc_fmri_connectomes_matched 


# === Define healthy-like subjects: non-demented (DEMENTED != 1) and no clinical diagnosis
healthy_like_ids = set(df_matched_adrc[
    (df_matched_adrc["DEMENTED"] != 1) | (df_matched_adrc["DEMENTED"].isna())
]["PTID"])

# === All valid subjects with connectomes and metadata (for inference)
valid_subjects = set(df_matched_adrc["PTID"])




# FA, VOLUME


##### FA

import torch

# === Get valid subjects: healthy ADRC subjects with connectome and metadata
valid_subjects = set(df_matched_adrc["PTID"])



# === Paths to stats files ===
fa_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/ADRC_Regional_Stats/studywide_stats_for_fa.txt"



df_fa = pd.read_csv(fa_path, sep="\t")

# === Remove ROI 0
df_fa = df_fa[df_fa["ROI"] != 0].reset_index(drop=True)

# === Select only columns for valid subjects
subject_cols_fa = [col for col in df_fa.columns if col in valid_subjects]
df_fa_transposed = df_fa[subject_cols_fa].transpose()

# === Set subject ID and convert to float
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)






#####VOLUME

# === Load Volume data ===
vol_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/ADRC_Regional_Stats/studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")



# === Remove ROI 0 row, as it is not a brain region of interest
df_vol = df_vol[df_vol["ROI"] != 0].reset_index(drop=True)

# === Select only columns corresponding to valid subjects
subject_cols_vol = [col for col in df_vol.columns if col in valid_subjects]
df_vol_transposed = df_vol[subject_cols_vol].transpose()

# === Set subject IDs as index and convert to float
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)





# === Combine FA and Volume into [84, 2] tensors per subject ===
multimodal_features_dict = {}

for subj in df_fa_transposed.index:
    if subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float32)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float32)

        if fa.shape[0] == 84 and vol.shape[0] == 84:  # safety check
            stacked = torch.stack([fa, vol], dim=1)  # shape: [84, 2]
            multimodal_features_dict[subj] = stacked
        else:
            print(f" Subject {subj} has unexpected feature shape: FA={fa.shape}, VOL={vol.shape}")

print(f" Subjects with valid multimodal node features: {len(multimodal_features_dict)}")

#check
# Print just the first subject key
first_subj = list(multimodal_features_dict.keys())[0]
print("First subject ID:", first_subj)

# Print shape and a snippet of its tensor
print("Feature shape:", multimodal_features_dict[first_subj].shape)
print("Feature preview:\n", multimodal_features_dict[first_subj][:5])  # print first 5 nodes




# === Normalize node features using only NoRisk + Familial
def normalize_multimodal_nodewise(feature_dict, healthy_like_ids):
    # Only use healthy-like tensors for mean/std
    healthy_tensors = [v for k, v in feature_dict.items() if k in healthy_like_ids]
    all_features = torch.stack(healthy_tensors)  # shape: [N_subjects, 84, 2]
    
    means = all_features.mean(dim=0)  # [84, 2]
    stds = all_features.std(dim=0) + 1e-8

    # Apply normalization to all subjects using healthy stats
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Apply
normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict, healthy_like_ids)






# === Matrix to graph function (ADRC) ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)  # upper triangle (excluding diagonal)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float32, device=device)

    # Get node features for this subject
    node_feats = node_features_dict[subject_id]  # shape: [84, 2]
    node_features = node_feats.to(device)

    return edge_index, edge_attr, node_features






# Print dti and fmri connectomes before log and th

import matplotlib.pyplot as plt
import seaborn as sns

# === Select subject to visualize ===
subject_id = list(adrc_dti_connectomes_matched.keys())[0]

# === Retrieve DTI and fMRI matrices (raw) ===
dti_matrix = adrc_dti_connectomes_matched[subject_id]
fmri_matrix = adrc_fmri_connectomes_matched[subject_id]

# === Retrieve age from metadata ===
age = df_matched_adrc[df_matched_adrc["PTID"] == subject_id]["SUBJECT_AGE_SCREEN"].values[0]

# === Plot side-by-side heatmaps (DTI + fMRI raw) ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(dti_matrix, cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"DTI (Raw) - {subject_id} (Age {age:.1f})")

plt.subplot(1, 2, 2)
sns.heatmap(fmri_matrix, cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"fMRI (Raw) - {subject_id} (Age {age:.1f})")

plt.tight_layout()
plt.show()

# === Print numeric values of DTI ===
print(f"\nNumeric DTI Matrix for subject {subject_id} (Age {age:.1f}):")
print(dti_matrix.to_numpy())

# === Print numeric values of fMRI ===
print(f"\nNumeric fMRI Matrix for subject {subject_id} (Age {age:.1f}):")
print(fmri_matrix.to_numpy())






#LOG AND THRESHOLD


# === Thresholding function ===
def threshold_connectome(matrix, percentile=95):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded = np.where(matrix_np >= threshold_value, matrix_np, 0)
    symmetrized = np.maximum(thresholded, thresholded.T)
    return pd.DataFrame(symmetrized, index=matrix.index, columns=matrix.columns)









#DTI 

# === Output dictionary ===
log_thresholded_connectomes_adrc_dti = {}

# === Track issues ===
invalid_shape_ids = []
failed_processing_ids = []

# === Apply to healthy subjects ===
for subject_id, matrix in adrc_dti_connectomes_matched.items():
    try:
        # Check shape (e.g., 84x84)
        if matrix.shape != (84, 84):
            invalid_shape_ids.append(subject_id)
            continue

        # Apply threshold and log1p
        thresholded = threshold_connectome(matrix, percentile=95)
        log_matrix = np.log1p(thresholded)
        log_thresholded_connectomes_adrc_dti[subject_id] = log_matrix

    except Exception as e:
        failed_processing_ids.append(subject_id)
        print(f" Error processing subject {subject_id}: {e}")

# === Summary ===

print(f" Total healthy DTI connectomes processed (threshold + log): {len(log_thresholded_connectomes_adrc_dti)}")


### log_thresholded_connectomes_adrc_dti_healthy = {}  -> dic Only healthy th and log connectomes dti adrc




#fMRI (NO LOG NEGATIVES...)

# === Output dictionary ===
log_thresholded_connectomes_adrc_fmri = {}

# === Track issues ===
invalid_shape_ids_fmri = []
failed_processing_ids_fmri = []

# === Apply to healthy subjects ===
for subject_id, matrix in adrc_fmri_connectomes_matched.items():
    try:
        if matrix.shape != (84, 84):
            invalid_shape_ids_fmri.append(subject_id)
            continue

        # Threshold + log transform
        thresholded = threshold_connectome(matrix, percentile=95)  # or maybe 90 for fMRI
        log_matrix = np.log1p(thresholded)
        log_thresholded_connectomes_adrc_fmri[subject_id] = log_matrix

    except Exception as e:
        failed_processing_ids_fmri.append(subject_id)
        print(f" Error processing subject {subject_id}: {e}")

print(f" Total healthy fMRI connectomes processed (threshold + log): {len(log_thresholded_connectomes_adrc_fmri)}")
print()




#PRINT AFTER LOG AND TH

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(log_thresholded_connectomes_adrc_dti[subject_id],
            cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"DTI (Log+TH) - {subject_id} (Age {age:.1f})")

plt.subplot(1, 2, 2)
sns.heatmap(log_thresholded_connectomes_adrc_fmri[subject_id],
            cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
plt.title(f"fMRI (Log+TH) - {subject_id} (Age {age:.1f})")

plt.tight_layout()
plt.show()



#Max and min values
import numpy as np

def get_upper_values(matrix_dict):
    values = []
    for mat in matrix_dict.values():
        arr = mat.values if isinstance(mat, pd.DataFrame) else mat
        upper = arr[np.triu_indices(arr.shape[0], k=1)]  # upper triangle without diagonal
        values.append(upper.flatten())
    return np.concatenate(values)

# === DTI RAW ===
dti_raw_vals = get_upper_values(adrc_dti_connectomes_matched)
print("DTI RAW")
print(f"Min: {dti_raw_vals.min():.2f} | Max: {dti_raw_vals.max():.2f} | Mean: {dti_raw_vals.mean():.2f} | Std: {dti_raw_vals.std():.2f}")
print()

# === fMRI RAW ===
fmri_raw_vals = get_upper_values(adrc_fmri_connectomes_matched)
print("fMRI RAW")
print(f"Min: {fmri_raw_vals.min():.2f} | Max: {fmri_raw_vals.max():.2f} | Mean: {fmri_raw_vals.mean():.2f} | Std: {fmri_raw_vals.std():.2f}")
print()

# === DTI LOG+TH ===
dti_log_vals = get_upper_values(log_thresholded_connectomes_adrc_dti)
print("DTI (Log+TH)")
print(f"Min: {dti_log_vals.min():.2f} | Max: {dti_log_vals.max():.2f} | Mean: {dti_log_vals.mean():.2f} | Std: {dti_log_vals.std():.2f}")
print()

# === fMRI LOG+TH (if available) ===
fmri_log_vals = get_upper_values(log_thresholded_connectomes_adrc_fmri)
print("fMRI (Log+TH)")
print(f"Min: {fmri_log_vals.min():.2f} | Max: {fmri_log_vals.max():.2f} | Mean: {fmri_log_vals.mean():.2f} | Std: {fmri_log_vals.std():.2f}")



# GRAPH METRICS

import networkx as nx
import numpy as np

def compute_clustering_coefficient(matrix):
    G = nx.from_numpy_array(matrix.to_numpy()) #graph
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

def compute_global_efficiency(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.global_efficiency(G)

def compute_local_efficiency(matrix):
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.local_efficiency(G)

#DTI
# === Ensure the metadata DataFrame has columns for the graph metrics ===
# These metrics will be filled per subject using their log-transformed connectome
df_matched_adrc["dti_Clustering_Coeff"] = np.nan
df_matched_adrc["dti_Path_Length"] = np.nan
df_matched_adrc["dti_Global_Efficiency"] = np.nan
df_matched_adrc["dti_Local_Efficiency"] = np.nan

# === Loop through each subject and compute graph metrics ===
for subject, matrix_log in log_thresholded_connectomes_adrc_dti.items():
    try:
        # Compute weighted clustering coefficient (averaged across nodes)
        dti_clustering = compute_clustering_coefficient(matrix_log)

        # Compute average shortest path length (weighted, with inverse distance)
        dti_path = compute_path_length(matrix_log)

        # Compute global efficiency (inverse of average shortest path over all node pairs)
        dti_global_eff = compute_global_efficiency(matrix_log)

        # Compute local efficiency (efficiency of each node's neighborhood)
        dti_local_eff = compute_local_efficiency(matrix_log)

        # Fill the computed values into the corresponding row in the metadata DataFrame
        df_matched_adrc.loc[df_matched_adrc["PTID"] == subject, [
            "dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"
        ]] = [dti_clustering, dti_path, dti_global_eff, dti_local_eff]

    except Exception as e:
        # Catch and report errors (e.g., disconnected graphs or NaNs)
        print(f" Failed to compute metrics for subject {subject}: {e}")

print()



#Functions to not use negative values in path, global and local efficiency

def clean_matrix(matrix):
    arr = matrix.to_numpy().copy()
    arr[arr <= 0] = 0  # Set negative and zero weights to 0
    return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


def compute_path_length_noneg(matrix):
    matrix = clean_matrix(matrix)  # clean before graph creation
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


def compute_global_efficiency_noneg(matrix):
    matrix = clean_matrix(matrix)
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.global_efficiency(G)


def compute_local_efficiency_noneg(matrix):
    matrix = clean_matrix(matrix)
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]
    return nx.local_efficiency(G)


#FMRI (using no log no th)

# === Add empty columns for fMRI graph metrics in the metadata DataFrame ===
df_matched_adrc["fmri_Clustering_Coeff"] = np.nan
df_matched_adrc["fmri_Path_Length"] = np.nan
df_matched_adrc["fmri_Global_Efficiency"] = np.nan
df_matched_adrc["fmri_Local_Efficiency"] = np.nan

# === Loop through each subject and compute graph metrics from fMRI connectomes ===
for subject, matrix_fmri in adrc_fmri_connectomes_matched.items():
    try:
        # Compute weighted clustering coefficient (averaged across nodes)
        fmri_clustering = compute_clustering_coefficient(matrix_fmri)

        # Compute average shortest path length (using inverse of weights)
        fmri_path = compute_path_length_noneg(matrix_fmri)

        # Compute global efficiency
        fmri_global_eff = compute_global_efficiency_noneg(matrix_fmri)

        # Compute local efficiency
        fmri_local_eff = compute_local_efficiency_noneg(matrix_fmri)

        # Fill computed metrics into the DataFrame
        df_matched_adrc.loc[df_matched_adrc["PTID"] == subject, [
            "fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"
        ]] = [fmri_clustering, fmri_path, fmri_global_eff, fmri_local_eff]

    except Exception as e:
        print(f"Failed to compute fMRI metrics for subject {subject}: {e}")







#Metadata


#Encode sex and apoe
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# === Encode SUBJECT_SEX directly (1 and 2) ===
# Optional: map 1 → 0 and 2 → 1 to start from 0
df_matched_adrc["sex_encoded"] = df_matched_adrc["SUBJECT_SEX"].map({1: 0, 2: 1})

# === Encode APOE (e.g., "3/4", "4/4", "2/3") ===
# Convert to string in case there are numbers
df_matched_adrc["genotype"] = LabelEncoder().fit_transform(df_matched_adrc["APOE"].astype(str))





# ADD BIOMARKERS
# Select biomarkers
biomarker_cols = ["AB40", "AB42", "TTAU", "PTAU181", "NFL", "GFAP"]


# Step 1: Convert AB40 and AB42 to numeric
df_matched_adrc["AB40"] = pd.to_numeric(df_matched_adrc["AB40"], errors='coerce')
df_matched_adrc["AB42"] = pd.to_numeric(df_matched_adrc["AB42"], errors='coerce')

# Step 2: Compute ratio
df_matched_adrc["AB_ratio"] = df_matched_adrc["AB42"] / df_matched_adrc["AB40"]
df_matched_adrc["AB_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Step 3: Include AB_ratio
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]

# Step 4: Compute mean/std from healthy-like only
means = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), biomarker_cols].mean()
stds = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), biomarker_cols].std() + 1e-8

# Step 5: Normalize all subjects using those stats
df_matched_adrc[biomarker_cols] = (df_matched_adrc[biomarker_cols] - means) / stds

# Step 6: Fill missing values
df_matched_adrc[biomarker_cols] = df_matched_adrc[biomarker_cols].fillna(-10)






#graph metrics

from scipy.stats import zscore

# === Define which columns are DTI graph-level metrics ===
dti_metrics = ["dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"]

# === Define which columns are fMRI graph-level metrics ===
fmri_metrics = ["fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"]


# === Compute means and stds using healthy-like only
means_dti = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), dti_metrics].mean()
stds_dti = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), dti_metrics].std() + 1e-8

means_fmri = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), fmri_metrics].mean()
stds_fmri = df_matched_adrc.loc[df_matched_adrc["PTID"].isin(healthy_like_ids), fmri_metrics].std() + 1e-8

# === Apply z-score normalization to all subjects using those stats
df_matched_adrc[dti_metrics] = (df_matched_adrc[dti_metrics] - means_dti) / stds_dti
df_matched_adrc[fmri_metrics] = (df_matched_adrc[fmri_metrics] - means_fmri) / stds_fmri





#Build global feature tensors

import torch

# === Demographic tensor per subject: [sex_encoded, genotype] ===
subject_to_demographic_tensor = {
    row["PTID"]: torch.tensor([
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in df_matched_adrc.iterrows()
}

# === DTI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_dti_graphmetrics_tensor = {
    row["PTID"]: torch.tensor([
        row["dti_Clustering_Coeff"],
        row["dti_Path_Length"],
        row["dti_Global_Efficiency"],
        row["dti_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_matched_adrc.iterrows()
}

# === fMRI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_fmri_graphmetrics_tensor = {
    row["PTID"]: torch.tensor([
        row["fmri_Clustering_Coeff"],
        row["fmri_Path_Length"],
        row["fmri_Global_Efficiency"],
        row["fmri_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_matched_adrc.iterrows()
}




# Step 5: Rebuild tensor dictionary
subject_to_biomarker_tensor = {
    row["PTID"]: torch.tensor(row[biomarker_cols].values.astype(np.float32))
    for _, row in df_matched_adrc.iterrows()
}



#Convert ADRC DTI matrices to PyTorch Geometric graph objects

import torch
from torch_geometric.data import Data

# === Device setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




# 1 -> DTI

# === Create list to store graph data objects
graph_data_list_adrc_dti = []


# === Create mapping: subject ID → age
subject_to_age = df_matched_adrc.set_index("PTID")["SUBJECT_AGE_SCREEN"].to_dict()



# === Iterate over each matched subject's processed matrix ===
for subject, matrix_log in log_thresholded_connectomes_adrc_dti.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_dti_graphmetrics_tensor:
            continue
        if subject not in subject_to_biomarker_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # === Convert connectome matrix to edge_index, edge_attr, node_features
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device, subject, normalized_node_features_dict
        )

        if subject not in subject_to_age:
            continue
        age = torch.tensor([subject_to_age[subject]], dtype=torch.float)


        # === Concatenate demographic + graph metrics to form global features
        demo_tensor = subject_to_demographic_tensor[subject]   # [2]
        dti_tensor = subject_to_dti_graphmetrics_tensor[subject]     # [4]
        biomarker_tensor = subject_to_biomarker_tensor[subject]    # [7]
        
        global_feat_dti = torch.cat([demo_tensor, dti_tensor,  biomarker_tensor], dim=0)  # [2+4=6]


        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_dti.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adrc_dti.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")

# === Preview one example graph ===
sample_graph = graph_data_list_adrc_dti[0]
print("=== ADRC Sample Graph ===")
print(f"Node feature shape (x): {sample_graph.x.shape}")         
print(f"Edge index shape: {sample_graph.edge_index.shape}")     
print(f"Edge attr shape: {sample_graph.edge_attr.shape}")       
print(f"Global features shape: {sample_graph.global_features.shape}")  
print(f"Target age (y): {sample_graph.y.item()}")                

print("\nFirst 5 edge weights:")
print(sample_graph.edge_attr[:5])

print("\nGlobal features vector:")
print(sample_graph.global_features)
print()



import matplotlib.pyplot as plt

ages = [data.y.item() for data in graph_data_list_adrc_dti]
plt.hist(ages, bins=20)
plt.title("Distribution of Real Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.show()



# 2 -> fMRI

# === Create list to store graph data objects
graph_data_list_adrc_fmri = []


# === Create mapping: subject ID → age
subject_to_age = df_matched_adrc.set_index("PTID")["SUBJECT_AGE_SCREEN"].to_dict()



# === Iterate over each matchedy subject's processed matrix ===
for subject, matrix_log in adrc_fmri_connectomes_matched.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        
        if subject not in subject_to_fmri_graphmetrics_tensor:
            continue
        
        if subject not in subject_to_biomarker_tensor:
            continue
        
        if subject not in normalized_node_features_dict:
            continue

        # === Convert connectome matrix to edge_index, edge_attr, node_features
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device, subject, normalized_node_features_dict
        )

        if subject not in subject_to_age:
            continue
        age = torch.tensor([subject_to_age[subject]], dtype=torch.float)


        # === Concatenate demographic + graph metrics to form global features
        demo_tensor = subject_to_demographic_tensor[subject]   # [2]
        fmri_tensor = subject_to_fmri_graphmetrics_tensor[subject]     # [4]
        biomarker_tensor = subject_to_biomarker_tensor[subject]    # [7]
        
        
        global_feat_fmri = torch.cat([demo_tensor, fmri_tensor, biomarker_tensor], dim=0)  # [2+4=6]


        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_fmri.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adrc_fmri.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")




# === Region names (84) ===
region_names = [
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




# 1) AGE

# 1.1 TOP 10 beeswarm colored by age group (young middle old)



import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load ADRC metadata containing subject age
metadata_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/metadata/alex-badea_2024-06-14 (copy).xlsx"
df_meta = pd.read_excel(metadata_path)

# Ensure subject IDs match the format used in SHAP CSVs
df_meta["PTID"] = df_meta["PTID"].astype(str).str.strip()

# === Folder containing edge SHAP CSVs
shap_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/shap_outputs"

# === Load SHAP CSVs and attach subject metadata (age + age group)
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_dti_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        subj_id = fname.split("_")[-1].replace(".csv", "")
        df["subject"] = subj_id

        # Match subject metadata
        age_row = df_meta[df_meta["PTID"] == subj_id]
        if not age_row.empty:
            age = age_row["SUBJECT_AGE_SCREEN"].values[0]
            df["age"] = age

            # Define age group
            if age < 45:
                df["group"] = "Young"
            elif age < 65:
                df["group"] = "Middle"
            else:
                df["group"] = "Old"

            shap_dfs.append(df)

# === Combine all subjects into a single DataFrame
all_df = pd.concat(shap_dfs, ignore_index=True)

# === Region names for brain regions (length = 84)
region_names = [...]  # Replace with your full list of 84 brain region names

# === Create edge label using region names
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

# === Compute top 10 most important edges based on mean absolute SHAP value
top_edges = (
    all_df.groupby("edge_label")["SHAP_value"]
    .apply(lambda x: x.abs().mean())
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# === Filter SHAP values for the top 10 edges
top_df = all_df[all_df["edge_label"].isin(top_edges)]

# === Sort edges for clean y-axis order in the plot
edge_order = top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

# === Create output folder for beeswarm plot
save_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/SHAP edges/beeswarm"
os.makedirs(save_dir, exist_ok=True)

# === Plot beeswarm with hue by age group
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

# === Save the plot
plt.savefig(os.path.join(save_dir, "top10_dti_edges_beeswarm_by_age.png"), dpi=300)
plt.close()

print("Saved beeswarm plot to:", save_dir)





# 1.2 Three beeswarms, by age

import os
import matplotlib.pyplot as plt
import seaborn as sns

# === Define output directory for saving plots
save_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/SHAP edges/beeswarm"
os.makedirs(save_dir, exist_ok=True)

# === Loop through each age group
for group in ["Young", "Middle", "Old"]:
    group_df = all_df[all_df["group"] == group].copy()

    # Skip if no subjects in this group
    if group_df.empty:
        print(f" No data for group: {group}")
        continue

    # === Identify top 10 most important edges for this group
    group_top_edges = (
        group_df.groupby("edge_label")["SHAP_value"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    # === Filter and sort for plotting
    group_top_df = group_df[group_df["edge_label"].isin(group_top_edges)]
    edge_order = group_top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

    print(f" {group}: {len(group_top_df)} rows from top 10 edges")

    # === Create strip plot (beeswarm)
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

    # === Save plot
    fname = f"top10_dti_edges_beeswarm_{group.lower()}.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=300)
    print(f" Saved: {out_path}")

    plt.close()






# 2) RISK GROUP
# 2.1 TOP 10 beeswarm colored by RISK group 

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Path where SHAP edge CSVs are stored
shap_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/shap_outputs"

# === Load ADRC metadata (should contain PTID, NORMCOG, DEMENTED, IMPNOMCI)
metadata_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/metadata/alex-badea_2024-06-14 (copy).xlsx" 
df_meta = pd.read_excel(metadata_path)

# Standardize subject ID
df_meta["PTID"] = df_meta["PTID"].astype(str).str.strip()

# Define group assignment logic
def infer_risk_group(row):
    if row["DEMENTED"] == 1:
        return "AD"
    elif row["IMPNOMCI"] == 1:
        return "MCI"
    elif row["NORMCOG"] == 1:
        return "NoRisk"
    else:
        return "Unknown"

df_meta["RiskGroup"] = df_meta.apply(infer_risk_group, axis=1)

# === Load SHAP CSVs and assign metadata
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_dti_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        subj_id = fname.split("_")[-1].replace(".csv", "")
        df["subject"] = subj_id

        subj_row = df_meta[df_meta["PTID"] == subj_id]
        if not subj_row.empty:
            df["age"] = subj_row["SUBJECT_AGE_SCREEN"].values[0]
            df["group"] = subj_row["RiskGroup"].values[0]
        else:
            df["age"] = None
            df["group"] = "Unknown"

        shap_dfs.append(df)

# Combine SHAP data
all_df = pd.concat(shap_dfs, ignore_index=True)

# Create edge labels from region names
all_df["edge_label"] = all_df.apply(
    lambda row: f"{region_names[int(row['Node_i'])]} ↔ {region_names[int(row['Node_j'])]}", axis=1
)

# Get top 10 edges by mean absolute SHAP value
top_edges = (
    all_df.groupby("edge_label")["SHAP_value"]
    .apply(lambda x: x.abs().mean())
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# Filter and sort for plotting
top_df = all_df[all_df["edge_label"].isin(top_edges)]
edge_order = top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

# Plot beeswarm
plt.figure(figsize=(12, 8))
sns.stripplot(
    data=top_df,
    x="SHAP_value",
    y="edge_label",
    hue="group",  # NoRisk, MCI, AD
    order=edge_order,
    dodge=True,
    jitter=0.25,
    alpha=0.7
)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("SHAP Value (impact on predicted brain age)")
plt.ylabel("Edge (connection)")
plt.title("Top 10 SHAP-Important Edges — Colored by Risk Group")
plt.legend(title="Group")
plt.tight_layout()
plt.show()

# === Output directory (create if needed)
save_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/SHAP edges/beeswarm"
os.makedirs(save_dir, exist_ok=True)

# === Save the plot
plt.savefig(os.path.join(save_dir, "top10_edges_beeswarm_by_risk_group.png"), dpi=300)





# 2.2 beeswarms by risk group



import os
import matplotlib.pyplot as plt
import seaborn as sns

# === Infer Risk Group from metadata if no "Risk" column exists
def infer_risk_group(row):
    if row["DEMENTED"] == 1:
        return "AD"
    elif row["IMPNOMCI"] == 1:
        return "MCI"
    elif row["NORMCOG"] == 1:
        return "NoRisk"
    else:
        return "Unknown"

# Apply inference to metadata
df_meta["RiskGroup"] = df_meta.apply(infer_risk_group, axis=1)

# === Merge RiskGroup into SHAP dataframe using subject ID
all_df = all_df.merge(
    df_meta[["PTID", "RiskGroup"]],
    left_on="subject",
    right_on="PTID",
    how="left"
)

all_df["RiskGroup"] = all_df["RiskGroup"].fillna("Unknown")

# === Output folder
save_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/SHAP edges/beeswarm_by_risk_separate"
os.makedirs(save_dir, exist_ok=True)

# === Loop through each risk group
for risk_group in all_df["RiskGroup"].unique():
    group_df = all_df[all_df["RiskGroup"] == risk_group].copy()

    if group_df.empty:
        print(f" No data for risk group: {risk_group}")
        continue

    # Compute top 10 most important edges for this group
    group_top_edges = (
        group_df.groupby("edge_label")["SHAP_value"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    # Filter and sort for plotting
    group_top_df = group_df[group_df["edge_label"].isin(group_top_edges)]
    edge_order = group_top_df.groupby("edge_label")["SHAP_value"].mean().sort_values().index

    print(f" {risk_group}: {len(group_top_df)} rows from top 10 edges")

    # === Create strip plot (beeswarm)
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

    # === Save plot
    fname = f"top10_edges_beeswarm_{risk_group.lower()}.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=300)
    print(f" Saved: {out_path}")

    plt.close()
