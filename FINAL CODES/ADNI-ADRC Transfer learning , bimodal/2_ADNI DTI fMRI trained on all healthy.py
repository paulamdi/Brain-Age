#ADNI DTI FMRI

#BEST 1

# ADNI Data

#Using 4 graph metrics
#Global feats normalized
#Multihead
#Edges

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

###################### Connectomes ############################
print("ADNI CONNECTOMES dti")


import os
import pandas as pd

# Define the base path where all subject visit folders are stored
base_path_adni_dti = "/home/bas/Desktop/Paula Pretraining/data/"

# Dictionary to store connectomes for each subject and timepoint
adni_connectomes_dti = {}

# Loop through every folder in the base directory
for folder_name in os.listdir(base_path_adni_dti):
    folder_path = os.path.join(base_path_adni_dti, folder_name)

    # Only process if the current item is a directory
    if os.path.isdir(folder_path):

        # Check if the folder name ends with '_connectomics'
        if "_connectomics" in folder_name:
            # Remove the suffix to get the subject ID and timepoint (e.g., R0072_y0)
            connectome_id = folder_name.replace("_connectomics", "")

            # The expected filename inside the folder (e.g., R0072_y0_onn_plain.csv)
            file_name = f"{connectome_id}_onn_plain.csv"
            file_path = os.path.join(folder_path, file_name)

            # Check if the expected file exists
            if os.path.isfile(file_path):
                try:
                    # Load the CSV as a DataFrame without headers
                    df = pd.read_csv(file_path, header=None)

                    # Store the matrix using ID as the key (e.g., "R0072_y0")
                    adni_connectomes_dti[connectome_id] = df

                except Exception as e:
                    # Handle any error during file loading
                    print(f"Error loading {file_path}: {e}")

# Summary: how many connectomes were successfully loaded
print("Total ADNI connectomes loaded:", len(adni_connectomes_dti))

# Show a few example keys from the dictionary
print("Example keys:", list(adni_connectomes_dti.keys())[:5])
print()



###################### Connectomes fmri ############################
print("ADNI CONNECTOMES fmri")


import os
import pandas as pd

# Define the base p# === LOAD fMRI CONNECTOMES (correlation matrices) ===
print("LOADING ADNI fMRI CONNECTOMES")

import os
import pandas as pd

base_path_fmri_adni = "/home/bas/Desktop/MyData/ADNI/data fMRI ADNI/functional_conn"  
adni_connectomes_fmri = {}

for filename in os.listdir(base_path_fmri_adni):
    if filename.endswith(".csv") and filename.startswith("func_connectome_corr_"):
        # Extract subject key: e.g., "R0074_y0"
        key = filename.replace("func_connectome_corr_", "").replace(".csv", "")
        
        file_path = os.path.join(base_path_fmri_adni, filename)
        try:
            df = pd.read_csv(file_path, header=None)
            adni_connectomes_fmri[key] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

print(f"Total fMRI connectomes loaded: {len(adni_connectomes_fmri)}")
print("Example keys:", list(adni_connectomes_fmri.keys())[:5])





###################### Metadata #############################


print ("ADNI Metadata\n")

import pandas as pd

# Load metadata from Excel file
metadata_path_adni = "/home/bas/Desktop/Paula Pretraining/metadata/idaSearch_3_19_2025FINAL.xlsx"
df_adni_metadata = pd.read_excel(metadata_path_adni, sheet_name="METADATA")

# Show basic info
print("ADNI metadata loaded. Shape:", df_adni_metadata.shape)
print()




# ADD COLUMN WITH  CORRESPONDING CONNECTOME KEYS

# Extract the numeric part of the Subject ID (e.g., from "003_S_4288" → "4288")
df_adni_metadata["Subject_Num"] = df_adni_metadata["Subject ID"].str.extract(r"(\d{4})$")

# Define mapping from visit description to simplified code
visit_map = {
    "ADNI3 Initial Visit-Cont Pt": "y0",
    "ADNI3 Year 4 Visit": "y4"
}

# Map the Visit column to y0 / y4 codes
df_adni_metadata["Visit_Clean"] = df_adni_metadata["Visit"].map(visit_map)

# Remove rows with unknown or unneeded visit types
df_adni_metadata = df_adni_metadata[df_adni_metadata["Visit_Clean"].notnull()]

# Build the final connectome key for each row (e.g., "R4288_y0")
df_adni_metadata["connectome_key"] = "R" + df_adni_metadata["Subject_Num"] + "_" + df_adni_metadata["Visit_Clean"]




# KEEP ONLY ONE LINE FOR EACH SUBJECT (TWO TIMEPOINTS EACH)

# Drop duplicate connectome_key rows to keep only one per connectome
df_adni_metadata_unique = df_adni_metadata.drop_duplicates(subset="connectome_key").copy()

# Summary
print("\nTotal unique connectome keys:", df_adni_metadata_unique.shape[0])
print(df_adni_metadata_unique[["Subject ID", "Visit", "connectome_key"]].head())
print()





#DTI+ FMRI 
# Intersect subjects that have both DTI and fMRI connectomes
shared_keys_adni = sorted(
    list(set(adni_connectomes_dti.keys()) & set(adni_connectomes_fmri.keys()))
)

print(f"Total ADNI bimodal subjects: {len(shared_keys_adni)}")
print()





############# Match connectomes and metadata ##################

print("MATCHING CONNECTOMES WITH METADATA\n")

# === Filter metadata to include only subjects with both modalities ===
df_adni_bimodal_metadata = df_adni_metadata_unique[
    df_adni_metadata_unique["connectome_key"].isin(shared_keys_adni)
].copy()

print("Filtered bimodal metadata shape:", df_adni_bimodal_metadata.shape)
print()




#Filter DTI and fMRI connectomes to shared subjects only

# === Filter DTI connectomes to bimodal subjects only ===
adni_dti_bimodal = {
    key: adni_connectomes_dti[key]
    for key in shared_keys_adni
}

# === Filter fMRI connectomes to bimodal subjects only ===
adni_fmri_bimodal = {
    key: adni_connectomes_fmri[key]
    for key in shared_keys_adni
}



#df_adni_bimodal_metadata-> metadata with dti and fmri connectomes

#adni_dti_bimodal -> dti connectomes with both types of connectome and metadara
#adni_fmri_bimodal -> fmri connectomes with both types of connectome and metadara



#KEEP HEALTHY
# === Keep only healthy control (CN) subjects from bimodal metadata ===
df_adni_bimodal_healthy = df_adni_bimodal_metadata[
    df_adni_bimodal_metadata["Research Group"] == "CN"
].copy()

print("Number of healthy CN bimodal subjects:", df_adni_bimodal_healthy.shape[0])
print()
# === Get keys for healthy CN bimodal subjects ===
shared_keys_healthy = df_adni_bimodal_healthy["connectome_key"].tolist()

# === Filter connectomes again based on healthy subset ===
adni_dti_bimodal_healthy = {k: adni_dti_bimodal[k] for k in shared_keys_healthy}
adni_fmri_bimodal_healthy = {k: adni_fmri_bimodal[k] for k in shared_keys_healthy}


#df_adni_bimodal_healthy-> helathy metadata with dti and fmri connectomes

#adni_dti_bimodal_healthy -> healthy dti connectomes with both types of connectome and metadara
#adni_fmri_bimodal_healthy -> healthy fmri connectomes with both types of connectome and metadara







#CHECK BEFORE LOG AND TH ADNI
import matplotlib.pyplot as plt
import seaborn as sns

# === Pick a random healthy bimodal subject ===
row = df_adni_bimodal_healthy.sample(1).iloc[0]

# === Extract subject info ===
connectome_key = row["connectome_key"]
subject_id = row["Subject ID"]
age = row["Age"]

# === Get DTI and fMRI connectomes ===
dti_matrix = adni_dti_bimodal_healthy[connectome_key]
fmri_matrix = adni_fmri_bimodal_healthy[connectome_key]

# === Print info and raw data ===
print(f"Subject ID: {subject_id}")
print(f"Connectome Key: {connectome_key}")
print(f"Age: {age}")
print("\nDTI matrix (first 5 rows):")
print(dti_matrix.head())
print("\nFMRI matrix (first 5 rows):")
print(fmri_matrix.head())

# === Plot DTI and fMRI heatmaps side by side ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(dti_matrix, cmap="viridis", square=True, ax=axes[0])
axes[0].set_title(f"DTI Connectome\n{connectome_key} (Age: {age})")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Region")

sns.heatmap(fmri_matrix, cmap="viridis", square=True, ax=axes[1])
axes[1].set_title(f"fMRI Connectome\n{connectome_key} (Age: {age})")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Region")

plt.tight_layout()
plt.show()





####################### FA + VOLUME FEATURES FOR ADNI #############################

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import networkx as nx
from torch_geometric.data import Data

# === Get valid subjects (those with both connectome and metadata matched) ===
valid_subjects = set(df_adni_bimodal_healthy["connectome_key"])

# === Load FA data from TSV ===
fa_path = "/home/bas/Desktop/Paula Pretraining/UTF-8ADNI_Regional_Stats/ADNI_Regional_Stats/ADNI_studywide_stats_for_fa.txt"
df_fa = pd.read_csv(fa_path, sep="\t")[1:]
df_fa = df_fa[df_fa["ROI"] != "0"].reset_index(drop=True)
subject_cols_fa = [col for col in df_fa.columns if col in valid_subjects]
df_fa_transposed = df_fa[subject_cols_fa].transpose()
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"
df_fa_transposed = df_fa_transposed.astype(float)

# === Load Volume data ===
vol_path = "/home/bas/Desktop/Paula Pretraining/UTF-8ADNI_Regional_Stats/ADNI_Regional_Stats/ADNI_studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")[1:]
df_vol = df_vol[df_vol["ROI"] != "0"].reset_index(drop=True)
subject_cols_vol = [col for col in df_vol.columns if col in valid_subjects]
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

# === Combine FA and Volume for each subject into a tensor [84, 2] ===
multimodal_features_dict = {}
for subj in df_fa_transposed.index:
    if subj in df_vol_transposed.index:
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float)
        stacked = torch.stack([fa, vol], dim=1)
        multimodal_features_dict[subj] = stacked

# === Normalize node features across subjects (node-wise) ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}


normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)


# === Matrix to graph function ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)
    node_feats = node_features_dict[subject_id]
    node_features = 0.5 * node_feats.to(device)
    return edge_index, edge_attr, node_features




# === Threshold function ===
def threshold_connectome(matrix, percentile=95):
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]
    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)
    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)

# === Apply threshold and log transform ===
log_thresholded_connectomes_adni_dti = {}
for subject, matrix in adni_dti_bimodal_healthy.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes_adni_dti[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)


# log_thresholded_connectomes_adni_dti -> adni helathy log th


#CHECK AFTER LOG AND TH ADNI
import matplotlib.pyplot as plt
import seaborn as sns

# === Pick a random healthy bimodal subject ===
row = df_adni_bimodal_healthy.sample(1).iloc[0]

# === Extract subject info ===
connectome_key = row["connectome_key"]
subject_id = row["Subject ID"]
age = row["Age"]

# === Get DTI and fMRI connectomes ===
dti_matrix = log_thresholded_connectomes_adni_dti[connectome_key]
fmri_matrix = adni_fmri_bimodal_healthy[connectome_key]

# === Print info and raw data ===
print(f"Subject ID: {subject_id}")
print(f"Connectome Key: {connectome_key}")
print(f"Age: {age}")
print("\nDTI matrix (first 5 rows):")
print(dti_matrix.head())
print("\nFMRI matrix (first 5 rows):")
print(fmri_matrix.head())

# === Plot DTI and fMRI heatmaps side by side ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(dti_matrix, cmap="viridis", square=True, ax=axes[0])
axes[0].set_title(f"DTI Connectome\n{connectome_key} (Age: {age})")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Region")

sns.heatmap(fmri_matrix, cmap="viridis", square=True, ax=axes[1])
axes[1].set_title(f"fMRI Connectome\n{connectome_key} (Age: {age})")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Region")

plt.tight_layout()
plt.show()




# === Graph metric functions ===
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

# === Add metrics to metadata ===
df_healthy_adni = df_adni_bimodal_healthy.reset_index(drop=True)
df_healthy_adni["dti_Clustering_Coeff"] = np.nan
df_healthy_adni["dti_Path_Length"] = np.nan
df_healthy_adni["dti_Global_Efficiency"] = np.nan
df_healthy_adni["dti_Local_Efficiency"] = np.nan

for subject, matrix_log in log_thresholded_connectomes_adni_dti.items():
    try:
        clustering_dti = compute_clustering_coefficient(matrix_log)
        path_dti = compute_path_length(matrix_log)
        global_eff_dti = compute_global_efficiency(matrix_log)
        local_eff_dti = compute_local_efficiency(matrix_log)

        df_healthy_adni.loc[df_healthy_adni["connectome_key"] == subject, [
            "dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"
        ]] = [clustering_dti, path_dti, global_eff_dti, local_eff_dti]
    except Exception as e:
        print(f"Failed to compute metrics for subject {subject}: {e}")





#Functions to not use negative values in path, global and local efficiency FOR FMRI

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
df_healthy_adni["fmri_Clustering_Coeff"] = np.nan
df_healthy_adni["fmri_Path_Length"] = np.nan
df_healthy_adni["fmri_Global_Efficiency"] = np.nan
df_healthy_adni["fmri_Local_Efficiency"] = np.nan

# === Loop through each subject and compute graph metrics from fMRI connectomes ===
for subject, matrix_fmri in adni_fmri_bimodal_healthy.items():
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
        df_healthy_adni.loc[df_healthy_adni["connectome_key"] == subject, [
            "fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"
        ]] = [fmri_clustering, fmri_path, fmri_global_eff, fmri_local_eff]

    except Exception as e:
        print(f"Failed to compute fMRI metrics for subject {subject}: {e}")








#Metadata


#Encode sex and apoe
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# === Encode SUBJECT_SEX directly (1 and 2) ===
df_healthy_adni["sex_encoded"] = LabelEncoder().fit_transform(df_healthy_adni["Sex"].astype(str))

# === Encode APOE  ===

df_healthy_adni["genotype"] = LabelEncoder().fit_transform(
    df_healthy_adni["APOE A1"].astype(str) + "_" + df_healthy_adni["APOE A2"].astype(str)
)






#Nomalize graph metrics with zscore
# === Define which columns are dti graph-level metrics ===
dti_metrics = ["dti_Clustering_Coeff", "dti_Path_Length", "dti_Global_Efficiency", "dti_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_healthy_adni[dti_metrics] = df_healthy_adni[dti_metrics].apply(zscore)



# === Define graph metric columns for fMRI ===

fmri_metrics = ["fmri_Clustering_Coeff", "fmri_Path_Length", "fmri_Global_Efficiency", "fmri_Local_Efficiency"]

# === Apply z-score normalization across subjects ===
df_healthy_adni[fmri_metrics] = df_healthy_adni[fmri_metrics].apply(zscore)






#Build global feature tensors

import torch

# === Demographic tensor per subject: [sex_encoded, genotype] ===
# === Build global feature tensors ===
subject_to_demographic_tensor = {
    row["connectome_key"]: torch.tensor([
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in df_healthy_adni.iterrows()
}

# === DTI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_dti_graphmetrics_tensor = {
    row["connectome_key"]: torch.tensor([
        row["dti_Clustering_Coeff"],
        row["dti_Path_Length"],
        row["dti_Global_Efficiency"],
        row["dti_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_healthy_adni.iterrows()
}

# === fMRI graph metrics tensor: [Clustering, Path Length, Global Eff., Local Eff.] ===
subject_to_fmri_graphmetrics_tensor = {
    row["connectome_key"]: torch.tensor([
        row["fmri_Clustering_Coeff"],
        row["fmri_Path_Length"],
        row["fmri_Global_Efficiency"],
        row["fmri_Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in df_healthy_adni.iterrows()
}







#################  CONVERT MATRIX TO GRAPH  ################

#Convert ADNI DTI matrices to PyTorch Geometric graph objects

import torch
from torch_geometric.data import Data

# === Device setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




# 1 -> DTI

# === Create list to store graph data objects
graph_data_list_adni_dti = []


# === Create mapping: subject ID → age
subject_to_age = df_healthy_adni.set_index("connectome_key")["Age"].to_dict()



# === Iterate over each healthy subject's processed matrix ===
for subject, matrix_log in log_thresholded_connectomes_adni_dti.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_dti_graphmetrics_tensor:
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
        
        global_feat_dti = torch.cat([demo_tensor, dti_tensor], dim=0)  # [2+4=6]


        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_dti.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adni_dti.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")

# === Preview one example graph ===
sample_graph = graph_data_list_adni_dti[0]
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

ages = [data.y.item() for data in graph_data_list_adni_dti]
plt.hist(ages, bins=20)
plt.title("Distribution of Real Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True)
plt.show()






# 2 -> fMRI

# === Create list to store graph data objects
graph_data_list_adni_fmri = []


# === Create mapping: subject ID → age
subject_to_age = df_healthy_adni.set_index("connectome_key")["Age"].to_dict()



# === Iterate over each healthy subject's processed matrix ===
for subject, matrix_log in adni_fmri_bimodal_healthy.items():
    try:
        # Skip if required components are missing
        if subject not in subject_to_demographic_tensor:
            continue
        
        if subject not in subject_to_fmri_graphmetrics_tensor:
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
        
        global_feat_fmri = torch.cat([demo_tensor, fmri_tensor], dim=0)  # [2+4=6]


        # === Create graph object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat_fmri.unsqueeze(0)
        )
        data.subject_id = subject
        graph_data_list_adni_fmri.append(data)
        
        # DEBUG: print to verify
        print(f"ADDED → Subject: {subject} | Assigned Age: {age.item()}")


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")








# === Check consistency of ages between df_healthy_adni and graph_data_list_adni_dti ===

# Convert DF to lookup dict
age_lookup_df = df_healthy_adni.set_index("connectome_key")["Age"].to_dict()

print("\n=== AGE CONSISTENCY CHECK (First 10 subjects) ===")
for g in graph_data_list_adni_dti[:10]:
    subj = g.subject_id
    age_from_df = age_lookup_df.get(subj, None)
    age_from_graph = g.y.item()
    
    print(f"Subject: {subj}")
    print(f" → Age in DataFrame:   {age_from_df}")
    print(f" → Age in Graph Object: {age_from_graph}")
    print(f" → MATCH? {'✅' if abs(age_from_df - age_from_graph) < 1e-2 else '❌'}")
    print("-" * 40)




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


        # Final MLP after concatenating DTI + fMRI + global
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + 16+32+32 , 128),  # 128 DTI + 128 fMRI + metadata
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
        
        global_embed = torch.cat([meta_embed, graph_dti_embed, graph_fmri_embed ], dim=1)

        # === Fusion and prediction ===
        x = torch.cat([x_dti, x_fmri, global_embed], dim=1)
        out = self.fc(x)

        return out





# Create lookup dictionaries from subject_id to graph
dti_dict = {g.subject_id: g for g in graph_data_list_adni_dti}
fmri_dict = {g.subject_id: g for g in graph_data_list_adni_fmri}

# Keep only common subjects
common_subjects = sorted(set(dti_dict.keys()) & set(fmri_dict.keys()))

# Build aligned list of (DTI, fMRI) graph pairs
graph_data_list_adni_bimodal = [(dti_dict[pid], fmri_dict[pid]) for pid in common_subjects]

print(f"Total bimodal subjects: {len(graph_data_list_adni_bimodal)}")



#collate_fn=collate_bimodal to properly batch pairs of DTI and fMRI graphs during training and evaluation. 
#It ensures each modality is grouped separately into Batch objects for input to the model.

def collate_bimodal(batch):
    data_dti_list, data_fmri_list = zip(*batch)  # separa los pares
    return Batch.from_data_list(data_dti_list), Batch.from_data_list(data_fmri_list)




    
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Usamos el DataLoader de torch_geometric

def train(model, train_loader, optimizer, criterion):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize the total loss for the epoch

    # Iterate through the training data loader
    for data_dti, data_fmri in train_loader:
        data_dti = data_dti.to(device)  # Move DTI graph to GPU
        data_fmri = data_fmri.to(device)  # Move fMRI graph to GPU

        optimizer.zero_grad()  # Clear previous gradients

        # Forward pass through the model with both DTI and fMRI inputs
        output = model(data_dti, data_fmri).view(-1)

        # Compute loss using the target age (assumed same in both DTI/fMRI)-> the target, the age is the same in both
        loss = criterion(output, data_dti.y)

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights

        total_loss += loss.item()  # Accumulate batch loss

    return total_loss / len(train_loader)  # Return average loss for the epoch




def evaluate(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss

    with torch.no_grad():  # Disable gradient computation
        for data_dti, data_fmri in test_loader:
            # Move each modality batch to the device
            data_dti = data_dti.to(device)
            data_fmri = data_fmri.to(device)

            # Forward pass through the model
            output = model(data_dti, data_fmri).view(-1)

            # Compute loss using DTI target (same age for both modalities)
            loss = criterion(output, data_dti.y)

            total_loss += loss.item()

    return total_loss / len(test_loader)





import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Training parameters
epochs = 300
patience = 40
k = 7  # Folds
batch_size = 6
repeats_per_fold = 10








# Initialize loss tracking
all_train_losses_bimodal = []
all_test_losses_bimodal = []
all_early_stopping_epochs_bimodal = []

# Get subject IDs
graph_subject_ids_bimodal = [pair[0].subject_id for pair in graph_data_list_adni_bimodal]
df_filtered = df_healthy_adni[df_healthy_adni["connectome_key"].isin(graph_subject_ids_bimodal)].copy()
df_filtered = df_filtered.set_index("connectome_key").loc[graph_subject_ids_bimodal].reset_index()

# Create stratification bins for age
ages = df_filtered["Age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Main cross-validation loop
for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_adni_bimodal, age_bins)):
    print(f"\n--- Bimodal Fold {fold+1}/{k} ---")

    train_data = [graph_data_list_adni_bimodal[i] for i in train_idx]
    test_data = [graph_data_list_adni_bimodal[i] for i in test_idx]

    fold_train_losses = []
    fold_test_losses = []

    for repeat in range(repeats_per_fold):
        print(f'  > Repeat {repeat+1}/{repeats_per_fold}')
        early_stop_epoch = None

        seed_everything(42 + repeat)

        #Dataloaders with collate_fn
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_bimodal)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_bimodal)


        model = DualGATv2_EarlyFusion().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"bimodal_model_fold_{fold+1}_rep_{repeat+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_epoch = epoch + 1
                    print(f"    Early stopping triggered at epoch {early_stop_epoch}.")
                    break

            scheduler.step()

        # Save early stop epoch
        if early_stop_epoch is None:
            early_stop_epoch = epochs
        all_early_stopping_epochs_bimodal.append((fold + 1, repeat + 1, early_stop_epoch))

        fold_train_losses.append(train_losses)
        fold_test_losses.append(test_losses)

    all_train_losses_bimodal.append(fold_train_losses)
    all_test_losses_bimodal.append(fold_test_losses)








#################  LEARNING CURVE GRAPH (MULTIPLE REPEATS)  ################

plt.figure(figsize=(10, 6))

# Plot average learning curves across all repeats for each fold
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses_bimodal[fold][rep],
                 label=f'Train Loss - Fold {fold+1} Rep {rep+1}', linestyle='dashed', alpha=0.5)
        plt.plot(all_test_losses_bimodal[fold][rep],
                 label=f'Test Loss - Fold {fold+1} Rep {rep+1}', alpha=0.5)

plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curves - ADNI Bimodal Model  (All Repeats)")
plt.legend(loc="upper right", fontsize=7)
plt.grid(True)
plt.tight_layout()
plt.show()


# ==== LEARNING CURVE PLOT (MEAN ± STD) FOR BIMODAL MODEL ====

import numpy as np
import matplotlib.pyplot as plt

# Compute mean and std for each epoch across all folds and repeats
avg_train_bimodal = []
avg_test_bimodal = []

for epoch in range(epochs):
    epoch_train = []
    epoch_test = []
    for fold in range(k):
        for rep in range(repeats_per_fold):
            if epoch < len(all_train_losses_bimodal[fold][rep]):
                epoch_train.append(all_train_losses_bimodal[fold][rep][epoch])
                epoch_test.append(all_test_losses_bimodal[fold][rep][epoch])
    avg_train_bimodal.append((np.mean(epoch_train), np.std(epoch_train)))
    avg_test_bimodal.append((np.mean(epoch_test), np.std(epoch_test)))

# Unpack into arrays
train_mean, train_std = zip(*avg_train_bimodal)
test_mean, test_std = zip(*avg_test_bimodal)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(train_mean, label="Train Mean", color="blue")
plt.fill_between(range(epochs), np.array(train_mean) - np.array(train_std),
                 np.array(train_mean) + np.array(train_std), color="blue", alpha=0.3)

plt.plot(test_mean, label="Test Mean", color="orange")
plt.fill_between(range(epochs), np.array(test_mean) - np.array(test_std),
                 np.array(test_mean) + np.array(test_std), color="orange", alpha=0.3)

plt.xlabel("Epoch")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curve Bimodal (Mean ± Std Across All Folds/Repeats)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





#####################  PREDICTION & METRIC ANALYSIS ACROSS FOLDS/REPEATS  #####################
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# === Initialize storage ===
fold_mae_list_bimodal = []
fold_r2_list_bimodal = []
fold_rmse_list_bimodal = []

all_subject_ids_bimodal = []
all_y_true_bimodal = []
all_y_pred_bimodal = []
all_folds_bimodal = []
all_repeats_bimodal = []

# Recalcular SKFold con la misma lógica
skf_bimodal = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
ages_bimodal = df_filtered["Age"].to_numpy()
age_bins_bimodal = pd.qcut(ages_bimodal, q=5, labels=False)

for fold, (train_idx, test_idx) in enumerate(skf_bimodal.split(graph_data_list_adni_bimodal, age_bins_bimodal)):
    print(f'\n--- BIMODAL Evaluating Fold {fold+1}/{k} ---')

    test_data = [graph_data_list_adni_bimodal[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_bimodal)


    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = DualGATv2_EarlyFusion().to(device)
        model.load_state_dict(torch.load(f"bimodal_model_fold_{fold+1}_rep_{rep+1}.pt"))
        model.eval()

        subject_ids_repeat = []
        y_true_repeat = []
        y_pred_repeat = []

        with torch.no_grad():
            for data_dti, data_fmri in test_loader:
                data_dti = data_dti.to(device)
                data_fmri = data_fmri.to(device)


                output = model(data_dti, data_fmri).view(-1)

                y_pred_repeat.extend(output.cpu().tolist())
                y_true_repeat.extend(data_dti.y.cpu().tolist())
                subject_ids_repeat.extend(data_dti.subject_id)

        # Save metrics
        mae = mean_absolute_error(y_true_repeat, y_pred_repeat)
        r2 = r2_score(y_true_repeat, y_pred_repeat)
        rmse = mean_squared_error(y_true_repeat, y_pred_repeat, squared=False)

        repeat_maes.append(mae)
        repeat_r2s.append(r2)

        fold_mae_list_bimodal.append(mae)
        fold_r2_list_bimodal.append(r2)
        fold_rmse_list_bimodal.append(rmse)

        all_subject_ids_bimodal.extend(subject_ids_repeat)
        all_y_true_bimodal.extend(y_true_repeat)
        all_y_pred_bimodal.extend(y_pred_repeat)
        all_folds_bimodal.extend([fold + 1] * len(y_true_repeat))
        all_repeats_bimodal.extend([rep + 1] * len(y_true_repeat))

    print(f">> Fold {fold+1} | MAE: {np.mean(repeat_maes):.2f} ± {np.std(repeat_maes):.2f} | R²: {np.mean(repeat_r2s):.2f} ± {np.std(repeat_r2s):.2f}")

# === Final aggregate results ===
mae_mean = np.mean(fold_mae_list_bimodal)
mae_std = np.std(fold_mae_list_bimodal)

rmse_mean = np.mean(fold_rmse_list_bimodal)
rmse_std = np.std(fold_rmse_list_bimodal)

r2_mean = np.mean(fold_r2_list_bimodal)
r2_std = np.std(fold_r2_list_bimodal)

print("\n================== FINAL METRICS BIMODAL ==================")
print(f"Global MAE:  {mae_mean:.2f} ± {mae_std:.2f}")
print(f"Global RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}")
print(f"Global R²:   {r2_mean:.2f} ± {r2_std:.2f}")
print("===========================================================")

# === Save predictions to DataFrame ===
df_preds_bimodal = pd.DataFrame({
    "Subject_ID": all_subject_ids_bimodal,
    "Real_Age": all_y_true_bimodal,
    "Predicted_Age": all_y_pred_bimodal,
    "Fold": all_folds_bimodal,
    "Repeat": all_repeats_bimodal
})

# Outliers
df_outliers = df_preds_bimodal[df_preds_bimodal["Predicted_Age"] > 120]
print("\n=== OUTLIER PREDICTIONS (>120 years) ===")
print(df_outliers)

print("\n=== Random Sample of Predictions ===")
print(df_preds_bimodal.sample(10))

df_preds_bimodal["AbsError"] = abs(df_preds_bimodal["Real_Age"] - df_preds_bimodal["Predicted_Age"])
print("\n=== Best Predictions (Lowest Error) ===")
print(df_preds_bimodal.nsmallest(10, "AbsError"))

print("\n=== Worst Predictions (Highest Error) ===")
print(df_preds_bimodal.nlargest(10, "AbsError"))

# === Scatter plot ===
plt.figure(figsize=(8, 6))
plt.scatter(df_preds_bimodal["Real_Age"], df_preds_bimodal["Predicted_Age"],
            alpha=0.7, edgecolors='k', label="Predictions")

min_age = min(df_preds_bimodal["Real_Age"].min(), df_preds_bimodal["Predicted_Age"].min()) - 5
max_age = max(df_preds_bimodal["Real_Age"].max(), df_preds_bimodal["Predicted_Age"].max()) + 5
plt.plot([min_age, max_age], [min_age, max_age], color='red', linestyle='--', label="Ideal (y = x)")

plt.xlim(min_age, max_age)
plt.ylim(min_age, max_age)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Ages (Bimodal Model)")
plt.legend()

# === Metrics box ===
textstr = (
    f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}\n"
    f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
    f"R²:   {r2_mean:.2f} ± {r2_std:.2f}"
)

plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

plt.grid(True)
plt.tight_layout()
plt.show()






#Already evaluated


######## TRAINING WITH ALL DATA FOR PRETRAINING #############


from torch_geometric.loader import DataLoader

# === Full loader with all ADNI healthy subjects ===
train_loader = DataLoader(graph_data_list_adni_bimodal, batch_size=6, shuffle=True, collate_fn=collate_bimodal)

# Initialize model and optimizer
model = DualGATv2_EarlyFusion().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = torch.nn.SmoothL1Loss(beta=1)

# === Training loop ===
epochs = 150
train_losses = []

print("\n=== Full Training on ADNI Bimodal (No Validation) ===")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for data_dti, data_fmri in train_loader:
        data_dti = data_dti.to(device)
        data_fmri = data_fmri.to(device)

        optimizer.zero_grad()
        output = model(data_dti, data_fmri).view(-1)
        loss = criterion(output, data_dti.y.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

# === Save pretrained model ===
torch.save(model.state_dict(), "brainage_adni_bimodal_pretrained.pt")
print("\n Pretrained model saved as 'brainage_adni_bimodal_pretrained.pt'")


