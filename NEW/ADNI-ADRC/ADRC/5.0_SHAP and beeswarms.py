#SHAP and BEESWARMS global features


#!Same Normalization with only healthy , avoid dat leackage

# Normalize graph metrics using z-scores computed only from "NoRisk" + "Familial" subjects (healthy-like group)
# This avoids data leakage from at-risk subjects (e.g., AD/MCI) during model training or inference
# All subjects are normalized using the mean and std from the healthy-like reference group


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




#APPLY PRETRAINED MODEL



######################  DEFINE MODEL  #########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class DualGATv2_EarlyFusion(nn.Module):
    def __init__(self):
        super(DualGATv2_EarlyFusion, self).__init__()

        # === Node Embedding shared across modalities (assumes same node features for DTI/fMRI) ===
        self.node_embed = nn.Sequential(
            nn.Linear(2, 64),  # Assumes 2 node features (e.g., FA, Volume)
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # === GATv2 backbone for DTI ===
        self.gnn_dti_1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_1 = BatchNorm(128)

        self.gnn_dti_2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_2 = BatchNorm(128)

        self.gnn_dti_3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_3 = BatchNorm(128)

        self.gnn_dti_4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_dti_4 = BatchNorm(128)

        # === GATv2 backbone for fMRI ===
        self.gnn_fmri_1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_1 = BatchNorm(128)

        self.gnn_fmri_2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_2 = BatchNorm(128)

        self.gnn_fmri_3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_3 = BatchNorm(128)

        self.gnn_fmri_4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn_fmri_4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.3)

        # === GLOBAL FEATURE BRANCHES (shared) ===
        self.meta_head = nn.Sequential(
            nn.Linear(2, 16),  # sex, genotype
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.graph_dti_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.graph_fmri_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        
        # Biomarkers: 6 features
        self.bio_head = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
          )




        # Final MLP after concatenating DTI + fMRI + global
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 16+32+32+32 , 128),  # 128 DTI + 128 fMRI + metadata + graph metrics dti + graph metrics fmri+  biomarkers
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data_dti, data_fmri):
        # === Node features shared ===
        x_dti = self.node_embed(data_dti.x)
        x_fmri = self.node_embed(data_fmri.x)

        # === DTI Stream ===
        x_dti = self.gnn_dti_1(x_dti, data_dti.edge_index, data_dti.edge_attr)
        x_dti = self.bn_dti_1(x_dti)
        x_dti = F.relu(x_dti)

        x_dti = F.relu(self.bn_dti_2(self.gnn_dti_2(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = F.relu(self.bn_dti_3(self.gnn_dti_3(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = F.relu(self.bn_dti_4(self.gnn_dti_4(x_dti, data_dti.edge_index, data_dti.edge_attr)) + x_dti)
        x_dti = self.dropout(x_dti)
        x_dti = global_mean_pool(x_dti, data_dti.batch)

        # === fMRI Stream ===
        x_fmri = self.gnn_fmri_1(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)
        x_fmri = self.bn_fmri_1(x_fmri)
        x_fmri = F.relu(x_fmri)

        x_fmri = F.relu(self.bn_fmri_2(self.gnn_fmri_2(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = F.relu(self.bn_fmri_3(self.gnn_fmri_3(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = F.relu(self.bn_fmri_4(self.gnn_fmri_4(x_fmri, data_fmri.edge_index, data_fmri.edge_attr)) + x_fmri)
        x_fmri = self.dropout(x_fmri)
        x_fmri = global_mean_pool(x_fmri, data_fmri.batch)

        # === Global features (same for both) ===
        global_feat = torch.cat([data_dti.global_features, data_fmri.global_features[:, 2:]], dim=1).to(data_dti.x.device).squeeze(1)  #.to(data_dti.x.device) makes sure global_feat is moved to the same device as input tensors

        meta_embed = self.meta_head(global_feat[:, 0:2]) # all rows, first two columns
        graph_dti_embed = self.graph_dti_head(global_feat[:, 2:6]) # all rows, columns from 3 to 6 (dti_Clustering, dti_PathLength, dti_GlobalEff, dti_LocalEff )
        graph_fmri_embed = self.graph_fmri_head(global_feat[:, 6:10])  # all rows, columns from 7 to 10  (fmri_Clustering, fmri_PathLength, fmri_GlobalEff, fmri_LocalEff)
        bio_embed = self.bio_head(global_feat[:, 10:17])     # 7 biomarkers

        
        
        
        global_embed = torch.cat([meta_embed, graph_dti_embed, graph_fmri_embed,  bio_embed ], dim=1)

        # === Fusion and prediction ===
        x = torch.cat([x_dti, x_fmri, global_embed], dim=1)
        out = self.fc(x)

        return out




# === Create subject-to-graph dictionaries for all subjects (not just healthy) ===
dti_dict_all = {g.subject_id: g for g in graph_data_list_adrc_dti}
fmri_dict_all = {g.subject_id: g for g in graph_data_list_adrc_fmri}

# === Create bimodal subject list for inference (only those with both DTI and fMRI) ===
common_subjects_all = sorted(set(dti_dict_all) & set(fmri_dict_all))
graph_data_list_adrc_bimodal_all = [(dti_dict_all[pid], fmri_dict_all[pid]) for pid in common_subjects_all]




# === Define custom collate function for bimodal input (DTI and fMRI graphs as separate batches) ===
from torch_geometric.data import Batch

def collate_bimodal(batch):
    data_dti_list, data_fmri_list = zip(*batch)
    return Batch.from_data_list(data_dti_list), Batch.from_data_list(data_fmri_list)


from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# === Load the pretrained model (trained on all healthy subjects) ===
model = DualGATv2_EarlyFusion().to(device)
model.load_state_dict(torch.load("brainage_adrc_bimodal_pretrained.pt"))
model.eval()






#SHAP GLOBAL FEATURES

import shap
import torch
import pandas as pd

# === SHAP wrapper that uses only global features ===
class GlobalOnlyModel(torch.nn.Module):
    def __init__(self, original_model):
        super(GlobalOnlyModel, self).__init__()
        self.meta_head = original_model.meta_head
        self.graph_dti_head = original_model.graph_dti_head
        self.graph_fmri_head = original_model.graph_fmri_head
        self.bio_head = original_model.bio_head
        self.fc = original_model.fc

    def forward(self, global_feats):
        # Split features by group
        meta = global_feats[:, 0:2]         # Sex, Genotype
        graph_dti = global_feats[:, 2:6]    # DTI: 4 graph metrics
        graph_fmri = global_feats[:, 6:10]  # fMRI: 4 graph metrics
        bio = global_feats[:, 10:]          # Biomarkers: 7

        # Pass through respective MLPs
        meta_embed = self.meta_head(meta)
        dti_embed = self.graph_dti_head(graph_dti)
        fmri_embed = self.graph_fmri_head(graph_fmri)
        bio_embed = self.bio_head(bio)

        # Dummy GNN outputs
        dummy_dti_embed = torch.zeros(global_feats.size(0), 128).to(global_feats.device)
        dummy_fmri_embed = torch.zeros(global_feats.size(0), 128).to(global_feats.device)

        # Concatenate
        combined = torch.cat([dummy_dti_embed, dummy_fmri_embed,
                              meta_embed, dti_embed, fmri_embed, bio_embed], dim=1)

        return self.fc(combined)

# === Extract global features and metadata
global_feats = []
subject_ids = []
ages = []


for data_dti, data_fmri in graph_data_list_adrc_bimodal_all:
    # Get full global feature vector (13 from DTI + 4 from fMRI metrics)
    dti_global = data_dti.global_features.squeeze(0)  # shape [13]
    fmri_metrics = data_fmri.global_features.squeeze(0)[2:6]  # shape [4]

    full_global = torch.cat([dti_global, fmri_metrics], dim=0)  # shape [17]
    global_feats.append(full_global)

    subject_ids.append(data_dti.subject_id)
    ages.append(data_dti.y.item())





global_feats_tensor = torch.stack(global_feats).to(device)
print("Global features tensor shape:", global_feats_tensor.shape)  # should be [N, 17]







global_feats_tensor = torch.stack(global_feats).to(device)  # [N, 17]

# === Wrap and prepare the model
wrapped_model = GlobalOnlyModel(model).to(device)
wrapped_model.eval()

# === Apply SHAP (DeepExplainer)
explainer = shap.DeepExplainer(wrapped_model, global_feats_tensor)
shap_values = explainer.shap_values(global_feats_tensor)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# === Global feature names
feature_names = [
    "Sex", "Genotype",
    "DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff",
    "fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff",
    "AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"
]


# === Save SHAP values
df_shap = pd.DataFrame(shap_values, columns=feature_names)
df_shap["Subject_ID"] = subject_ids
df_shap["Age"] = ages
df_shap = df_shap[["Subject_ID", "Age"] + feature_names]

df_shap.to_csv("shap_global_features_adrc_bimodal.csv", index=False)
print("Saved: shap_global_features_adrc_bimodal.csv")




# 1 Beeswarm plot — ONE FOR ALL  global features

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load SHAP values (already saved to CSV)
df_shap = pd.read_csv("shap_global_features_adrc_bimodal.csv")

# === Define correct order of features (17 global features used in your model)
feature_names = [
    "Sex", "Genotype",
    "DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff",
    "fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff",
    "AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"
]


# === Extract SHAP matrix
shap_matrix = df_shap[feature_names].values

# === Create SHAP Explanation object
shap_values_all = shap.Explanation(
    values=shap_matrix,
    data=shap_matrix,  # optional for beeswarm
    feature_names=feature_names
)

# === Beeswarm plot
plt.figure(figsize=(12, 6))
shap.plots.beeswarm(shap_values_all, max_display=len(feature_names), show=False)
plt.title("ADRC bimodal SHAP Beeswarm — All Global Features (All Subjects)")
plt.tight_layout()
plt.show()





#2 Beeswarm plots — PER GROUP

# === Define grouped feature names (consistent with df_shap columns)
demographic_cols = ["Sex", "Genotype"]
graphmetric_dti_cols = ["DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff"]
graphmetric_fmri_cols = ["fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff"]
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]


# === Helper function for plotting
def plot_beeswarm(df, features, title):
    shap_matrix = df[features].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Plot each group
plot_beeswarm(df_shap, demographic_cols, "ADRC bimodal SHAP Beeswarm — Demographics")
plot_beeswarm(df_shap, graphmetric_dti_cols, "ADRC bimodal SHAP Beeswarm — DTI Graph Metrics")
plot_beeswarm(df_shap, graphmetric_fmri_cols, "ADRC bimodal SHAP Beeswarm — fMRI Graph Metrics")
plot_beeswarm(df_shap, biomarker_cols, "ADRC bimodal SHAP Beeswarm — Biomarkers")







#3 Beeswarm by age groups

import shap
import matplotlib.pyplot as plt

# === Load SHAP values CSV if needed
df_shap = pd.read_csv("shap_global_features_adrc_bimodal.csv")

# === Create age bins using tertiles
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# === Optional: check counts
print(df_shap["Age_Group"].value_counts())

# === Get correct feature names (excluding Subject_ID, Age, Age_Group)
exclude_cols = ["Subject_ID", "Age", "Age_Group"]
feature_names = [col for col in df_shap.columns if col not in exclude_cols]

# === Beeswarm plot function
def plot_beeswarm(df_group, title):
    shap_matrix = df_group[feature_names].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=feature_names
    )
    plt.figure(figsize=(10, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(feature_names), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Loop over age groups
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    plot_beeswarm(df_group, f"SHAP Beeswarm — Age Group: {group}")



#4 age and feature type

import shap
import pandas as pd
import matplotlib.pyplot as plt

# === Load SHAP values
df_shap = pd.read_csv("shap_global_features_adrc_bimodal.csv")

# === Create Age Groups (tertiles)
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# === Define feature groups (must match column names exactly)
demographic_cols = ["Sex", "Genotype"]
graphmetric_dti_cols = ["DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff"]
graphmetric_fmri_cols = ["fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff"]
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]

# === Beeswarm plotting function
def plot_beeswarm(df, features, title):
    shap_matrix = df[features].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Loop over age groups and plot each feature group
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]

    # Plot for each group of features
    plot_beeswarm(df_group, demographic_cols, f"ADRC Bimodal SHAP Beeswarm — Demographics ({group})")
    plot_beeswarm(df_group, graphmetric_dti_cols, f"ADRC Bimodal SHAP Beeswarm — DTI Graph Metrics ({group})")
    plot_beeswarm(df_group, graphmetric_fmri_cols, f"ADRC Bimodal SHAP Beeswarm — fMRI Graph Metrics ({group})")
    plot_beeswarm(df_group, biomarker_cols, f"ADRC Bimodal SHAP Beeswarm — Biomarkers ({group})")




#5 PERSONALIZED BY SUBJECT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load SHAP values for all subjects (with Age and Subject_ID) ===
df_shap = pd.read_csv("shap_global_features_adrc_bimodal.csv")

# === Create Age Groups using tertiles
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# === Define SHAP feature columns (exclude ID and Age)
exclude_cols = ["Subject_ID", "Age", "Age_Group"]
feature_names = [col for col in df_shap.columns if col not in exclude_cols]

# === Select one representative subject per age group
subject_young = df_shap.loc[df_shap["Age"].idxmin()]  # Youngest subject
subject_old = df_shap.loc[df_shap["Age"].idxmax()]    # Oldest subject

# For middle, choose subject closest to median age
median_age = df_shap["Age"].median()
subject_middle = df_shap.iloc[(df_shap["Age"] - median_age).abs().argsort().iloc[0]]

# === Plotting function: signed horizontal bar plot of SHAP values per subject
def plot_subject_shap_signed(subject_row):
    subject_id = subject_row["Subject_ID"]
    age = subject_row["Age"]

    # Extract SHAP values (signed)
    shap_values = subject_row[feature_names]

    # Sort by absolute value, preserve sign
    shap_sorted = shap_values.reindex(shap_values.abs().sort_values().index)

    # Plot
    plt.figure(figsize=(8, 5))
    shap_sorted.plot(kind="barh", color=shap_sorted.apply(lambda x: "crimson" if x < 0 else "steelblue"))
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
    plt.xlabel("SHAP value (contribution to prediction)")
    plt.title(f"ADRC Bimodal SHAP Global Feature Impact — Subject {subject_id} (Age: {int(age)})")
    plt.tight_layout()
    plt.show()

# === Apply to each selected subject
plot_subject_shap_signed(subject_young)
plot_subject_shap_signed(subject_middle)
plot_subject_shap_signed(subject_old)



#SAVE

import os
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load SHAP CSV
df_shap = pd.read_csv("shap_global_features_adrc_bimodal.csv")

# Define features
feature_names = [
    "Sex", "Genotype",
    "DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff",
    "fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff",
    "AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"
]
demographic_cols = ["Sex", "Genotype"]
graphmetric_dti_cols = ["DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff"]
graphmetric_fmri_cols = ["fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff"]
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]

# Make output directories
os.makedirs("beeswarm_plots/all_global", exist_ok=True)
os.makedirs("beeswarm_plots/by_group", exist_ok=True)
os.makedirs("beeswarm_plots/by_age_group", exist_ok=True)
os.makedirs("beeswarm_plots/by_age_and_type", exist_ok=True)
os.makedirs("beeswarm_plots/personalized", exist_ok=True)

# === Beeswarm #1 — All Global Features
shap_values_all = shap.Explanation(
    values=df_shap[feature_names].values,
    data=df_shap[feature_names].values,
    feature_names=feature_names
)
plt.figure(figsize=(12, 6))
shap.plots.beeswarm(shap_values_all, max_display=len(feature_names), show=False)
plt.title("ADRC bimodal SHAP Beeswarm — All Global Features")
plt.tight_layout()
plt.savefig("beeswarm_plots/all_global/beeswarm_all_global.png")
plt.close()

# === Beeswarm #2 — Per Feature Group
def plot_and_save(df, features, title, path):
    shap_exp = shap.Explanation(
        values=df[features].values,
        data=df[features].values,
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

plot_and_save(df_shap, demographic_cols, "Demographics", "beeswarm_plots/by_group/demographics.png")
plot_and_save(df_shap, graphmetric_dti_cols, "DTI Graph Metrics", "beeswarm_plots/by_group/dti_graph_metrics.png")
plot_and_save(df_shap, graphmetric_fmri_cols, "fMRI Graph Metrics", "beeswarm_plots/by_group/fmri_graph_metrics.png")
plot_and_save(df_shap, biomarker_cols, "Biomarkers", "beeswarm_plots/by_group/biomarkers.png")

# === Beeswarm #3 — By Age Group
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    plot_and_save(df_group, feature_names, f"SHAP Beeswarm — Age Group: {group}", f"beeswarm_plots/by_age_group/{group.lower()}.png")

# === Beeswarm #4 — Age Group + Feature Type
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    plot_and_save(df_group, demographic_cols, f"Demographics ({group})", f"beeswarm_plots/by_age_and_type/{group.lower()}_demographics.png")
    plot_and_save(df_group, graphmetric_dti_cols, f"DTI Graph Metrics ({group})", f"beeswarm_plots/by_age_and_type/{group.lower()}_dti.png")
    plot_and_save(df_group, graphmetric_fmri_cols, f"fMRI Graph Metrics ({group})", f"beeswarm_plots/by_age_and_type/{group.lower()}_fmri.png")
    plot_and_save(df_group, biomarker_cols, f"Biomarkers ({group})", f"beeswarm_plots/by_age_and_type/{group.lower()}_biomarkers.png")

# === Beeswarm #5 — Personalized per Subject
subject_young = df_shap.loc[df_shap["Age"].idxmin()]
subject_old = df_shap.loc[df_shap["Age"].idxmax()]
subject_middle = df_shap.iloc[(df_shap["Age"] - df_shap["Age"].median()).abs().argsort().iloc[0]]

def plot_subject(subject_row, filename):
    subject_id = subject_row["Subject_ID"]
    age = subject_row["Age"]
    shap_vals = subject_row[feature_names]
    shap_sorted = shap_vals.reindex(shap_vals.abs().sort_values().index)
    plt.figure(figsize=(8, 5))
    shap_sorted.plot(kind="barh", color=shap_sorted.apply(lambda x: "crimson" if x < 0 else "steelblue"))
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
    plt.xlabel("SHAP value (contribution to prediction)")
    plt.title(f"Subject {subject_id} (Age: {int(age)})")
    plt.tight_layout()
    plt.savefig(f"beeswarm_plots/personalized/{filename}")
    plt.close()

plot_subject(subject_young, "subject_young.png")
plot_subject(subject_middle, "subject_middle.png")
plot_subject(subject_old, "subject_old.png")
