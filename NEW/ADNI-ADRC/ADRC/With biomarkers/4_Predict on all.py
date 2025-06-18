#PREDICTING ON ALL RISKS

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






# === Normalize node features (across subjects, per node+modality) ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # shape: [N_subjects, 84, 2]
    means = all_features.mean(dim=0)  # shape: [84, 2]
    stds = all_features.std(dim=0) + 1e-8  # shape: [84, 2] to avoid div by zero
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)

print(f" Node features normalized. Example shape: {list(normalized_node_features_dict.values())[0].shape}")











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


# Step 1: Convert AB40 and AB42 to numeric (safe conversion)
df_matched_adrc["AB40"] = pd.to_numeric(df_matched_adrc["AB40"], errors='coerce')
df_matched_adrc["AB42"] = pd.to_numeric(df_matched_adrc["AB42"], errors='coerce')

# Step 2: Compute the ratio safely
df_matched_adrc["AB_ratio"] = df_matched_adrc["AB42"] / df_matched_adrc["AB40"]
df_matched_adrc["AB_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Include the new ratio along with existing biomarkers
biomarker_cols = ["AB40", "AB42", "AB_ratio", "TTAU", "PTAU181", "NFL", "GFAP"]


# Z-score valid values
df_matched_adrc[biomarker_cols] = df_matched_adrc[biomarker_cols].apply(
    lambda col: (col - col.mean()) / (col.std() + 1e-8)
)

# Fill NaNs with -10 (clear out-of-distribution marker)
df_matched_adrc[biomarker_cols] = df_matched_adrc[biomarker_cols].fillna(-10)











#Nomalize graph metrics with zscore
# === Define which columns are dti graph-level metrics ===
dti_metrics = ["dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_matched_adrc[dti_metrics] = df_matched_adrc[dti_metrics].apply(zscore)



# === Define graph metric columns for fMRI ===

fmri_metrics = ["fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_matched_adrc[fmri_metrics] = df_matched_adrc[fmri_metrics].apply(zscore)





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

# === Create DataLoader with all subjects for inference (using bimodal collate function) ===
loader = DataLoader(
    graph_data_list_adrc_bimodal_all,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_bimodal
)

# === Perform inference on all subjects ===
subject_ids = []
true_ages = []
predicted_ages = []

with torch.no_grad():
    for data_dti, data_fmri in loader:
        data_dti = data_dti.to(device)
        data_fmri = data_fmri.to(device)

        pred = model(data_dti, data_fmri).item()
        subject_ids.append(data_dti.subject_id[0])  # same for both modalities
        true_ages.append(data_dti.y.item())
        predicted_ages.append(pred)

# === Create prediction DataFrame ===
df_preds = pd.DataFrame({
    "Subject_ID": subject_ids,
    "Age": true_ages,
    "Predicted_Age": predicted_ages
})

# === Compute Brain Age Gap (BAG) ===
df_preds["BAG"] = df_preds["Predicted_Age"] - df_preds["Age"]

# === Correct BAG using linear regression to remove age bias: BAG ~ Age ===
reg = LinearRegression().fit(df_preds[["Age"]], df_preds["BAG"])
df_preds["cBAG"] = df_preds["BAG"] - reg.predict(df_preds[["Age"]])

# === Save predictions (age, predicted age, BAG, cBAG) ===
df_preds.to_csv("bimodal_brainage_predictions_adrc_all.csv", index=False)
print("Saved: bimodal_brainage_predictions_adrc_all.csv")

# === Optional: quick scatter plot of predictions vs real age ===
plt.figure(figsize=(7, 6))
plt.scatter(df_preds["Age"], df_preds["Predicted_Age"], edgecolors='k', alpha=0.6)
plt.plot([20, 100], [20, 100], 'r--')
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Age (All ADRC)")
plt.grid(True)
plt.tight_layout()
plt.show()





#CVS with  subj age pred age, bag, cbag, metadata
# === Ensure both ID columns are in string format for correct merge ===
df_matched_adrc["PTID"] = df_matched_adrc["PTID"].astype(str)
df_preds["Subject_ID"] = df_preds["Subject_ID"].astype(str)

# === Merge predictions with full metadata ===
df_preds_full = pd.merge(df_preds, df_matched_adrc, left_on="Subject_ID", right_on="PTID", how="left")

# === Optional: reorder columns to bring predictions to the front ===
cols_first = ["Subject_ID", "Age", "Predicted_Age", "BAG", "cBAG"]
other_cols = [col for col in df_preds_full.columns if col not in cols_first]
df_preds_full = df_preds_full[cols_first + other_cols]

# === Save final merged predictions with metadata ===
df_preds_full.to_csv("brainage_predictions_adrc_all_with_metadata.csv", index=False)
print("Saved: brainage_predictions_adrc_all_with_metadata.csv")
