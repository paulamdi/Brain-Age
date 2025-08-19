# ADNI Data

#Adni pretraining using:
    #Node features: FA, VOL​ and MD(zeros) , nodewiseclustering
    #Demografic features: Sex and genotype (label encoded)
    #Graph metrics: Only clustering coefficient and path lenght 
    #PCAs and syst and dias =0 to be similar to addecode
    


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
print("ADNI CONNECTOMES")


import os
import pandas as pd

# Define the base path where all subject visit folders are stored
base_path_adni = "/home/bas/Desktop/Paula Pretraining/data/"

# Dictionary to store connectomes for each subject and timepoint
adni_connectomes = {}

# Loop through every folder in the base directory
for folder_name in os.listdir(base_path_adni):
    folder_path = os.path.join(base_path_adni, folder_name)

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
                    adni_connectomes[connectome_id] = df

                except Exception as e:
                    # Handle any error during file loading
                    print(f"Error loading {file_path}: {e}")

# Summary: how many connectomes were successfully loaded
print("Total ADNI connectomes loaded:", len(adni_connectomes))

# Show a few example keys from the dictionary
print("Example keys:", list(adni_connectomes.keys())[:5])
print()



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


############# Match connectomes and metadata ##################

print("MATCHING CONNECTOMES WITH METADATA\n")

# Keep only metadata rows where the connectome key exists in the connectome dictionary
df_matched_adni = df_adni_metadata_unique[
    df_adni_metadata_unique["connectome_key"].isin(adni_connectomes.keys())
].copy()

# Show result
print("Matched connectomes:", df_matched_adni.shape[0])
print()


#Printing a connectome wit its age and subject id
 
import seaborn as sns
import matplotlib.pyplot as plt

# Select a row from the matched metadata (you can change the index)
row = df_matched_adni.sample(1).iloc[0]


# Extract subject info
subject_id = row["Subject ID"]
connectome_key = row["connectome_key"]
age = row["Age"]

# Get the connectome matrix
matrix = adni_connectomes[connectome_key]

# Print subject info and connectome
print(f"Subject ID: {subject_id}")
print(f"Connectome Key: {connectome_key}")
print(f"Age: {age}")
print("Connectome matrix (first 5 rows):")
print(matrix.head())
print()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="viridis", square=True)
plt.title(f"Connectome Heatmap - {connectome_key} (Age: {age})")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.show()


# Keeping healthy subjects

# Keep only healthy control subjects (CN)
df_matched_adni_healthy = df_matched_adni[df_matched_adni["Research Group"] == "CN"].copy()

# Show result
print("Number of healthy ADNI subjects with matched connectomes:", df_matched_adni_healthy.shape[0])
print()
print()




# === Filter connectomes to include only those from healthy controls (CN) ===
matched_connectomes_healthy_adni = {
    row["connectome_key"]: adni_connectomes[row["connectome_key"]]
    for _, row in df_matched_adni_healthy.iterrows()
}

print(f"Number of healthy ADNI connectomes selected: {len(matched_connectomes_healthy_adni)}")





# df_matched_adni:
# → Cleaned ADNI metadata matched to available connectomes
# → Includes all subjects (CN, MCI, AD)

# adni_connectomes:
# → Dictionary of all loaded ADNI connectome matrices
# → Key: subject ID with timepoint (e.g., R4288_y0)
# → Value: raw connectome matrix
# → Includes all subjects





# df_matched_adni_healthy:
# → Metadata of only healthy control (CN) subjects
# → Subset of df_matched_adni

# matched_connectomes_healthy_adni:
# → Connectomes of only healthy subjects (CN)
# → Subset of adni_connectomes (filtered to match df_matched_adni_healthy)










####################### FA + VOLUME FEATURES FOR ADNI #############################

import torch
import pandas as pd

# === Get valid subjects (those with both connectome and metadata matched) ===
valid_subjects = set(df_matched_adni_healthy["connectome_key"])

# === Load FA data from TSV ===
fa_path = "/home/bas/Desktop/Paula Pretraining/UTF-8ADNI_Regional_Stats/ADNI_Regional_Stats/ADNI_studywide_stats_for_fa.txt"
df_fa = pd.read_csv(fa_path, sep="\t")  # Load FA as tab-separated file

# Remove first row (header artifact) and any rows with ROI == 0
df_fa = df_fa[1:]
df_fa = df_fa[df_fa["ROI"] != "0"]
df_fa = df_fa.reset_index(drop=True)

# Keep only the subjects that match connectome+metadata keys
subject_cols_fa = [col for col in df_fa.columns if col in valid_subjects]

# Transpose the dataframe so that rows = subjects and columns = ROIs
df_fa_transposed = df_fa[subject_cols_fa].transpose()

# Rename columns to standard ROI labels
df_fa_transposed.columns = [f"ROI_{i+1}" for i in range(df_fa_transposed.shape[1])]
df_fa_transposed.index.name = "subject_id"

# Convert all data to float
df_fa_transposed = df_fa_transposed.astype(float)

# === Load Volume data (same structure and logic as FA) ===
vol_path = "/home/bas/Desktop/Paula Pretraining/UTF-8ADNI_Regional_Stats/ADNI_Regional_Stats/ADNI_studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")

# Clean volume data: remove first row and ROI==0 rows
df_vol = df_vol[1:]
df_vol = df_vol[df_vol["ROI"] != "0"]
df_vol = df_vol.reset_index(drop=True)

# Keep only columns corresponding to valid connectome subjects
subject_cols_vol = [col for col in df_vol.columns if col in valid_subjects]

# Transpose: rows = subjects, columns = ROIs
df_vol_transposed = df_vol[subject_cols_vol].transpose()
df_vol_transposed.columns = [f"ROI_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"

# Convert all values to float
df_vol_transposed = df_vol_transposed.astype(float)

# === Combine FA and Volume for each subject into a tensor [84, 2] ===
multimodal_features_dict = {}

# Iterate over subject IDs
for subj in df_fa_transposed.index:
    if subj in df_vol_transposed.index:
        # Get FA and Volume vectors
        fa = torch.tensor(df_fa_transposed.loc[subj].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed.loc[subj].values, dtype=torch.float)
        
        # Stack them along last dimension → shape [84, 2]
        stacked = torch.stack([fa, vol], dim=1)

        # Store in dictionary
        multimodal_features_dict[subj] = stacked

# === Normalize node features across subjects (node-wise) ===
def normalize_multimodal_nodewise(feature_dict):
    # Stack all subjects into a tensor [N_subjects, 84, 2]
    all_features = torch.stack(list(feature_dict.values()))
    
    # Compute mean and std for each node (ROI) across subjects
    means = all_features.mean(dim=0)  # → shape [84, 2]
    stds = all_features.std(dim=0) + 1e-8  # → shape [84, 2]

    # Normalize each subject individually
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Apply normalization
normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)





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

    clustering_dict = nx.clustering(G, weight="weight")
    clustering_values = [clustering_dict[i] for i in range(len(clustering_dict))]
    return torch.tensor(clustering_values, dtype=torch.float).unsqueeze(1)  # [84, 1]





############## THRESHOLD FUNCTION ##############

import numpy as np
import pandas as pd

def threshold_connectome(matrix, percentile=95):
    """
    Apply percentile-based thresholding to a connectome matrix.

    Parameters:
    - matrix (pd.DataFrame): The original connectome matrix (84x84).
    - percentile (float): The percentile threshold to keep.

    Returns:
    - thresholded_matrix (pd.DataFrame): Thresholded matrix with only strong connections.
    """
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)
    values = matrix_np[mask]

    threshold_value = np.percentile(values, 100 - percentile)
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)

    return pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)


########### APPLY THRESHOLD AND LOG TRANSFORMATION TO ADNI CONNECTOMES #####################

log_thresholded_connectomes_adni = {}

for subject, matrix in matched_connectomes_healthy_adni.items():
    # Step 1: Threshold top 5% of connections
    thresholded_matrix = threshold_connectome(matrix, percentile=95)

    # Step 2: Apply log(x + 1) transformation
    log_matrix = np.log1p(thresholded_matrix)

    # Step 3: Store the transformed matrix
    log_thresholded_connectomes_adni[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)

# === Visual check of the first transformed matrix ===
for subject, matrix_log in list(log_thresholded_connectomes_adni.items())[:1]:
    print(f"Log-transformed connectome (ADNI) for subject {subject}:")
    print(matrix_log.head())
    print()
    

##################### MATRIX TO GRAPH #######################

import torch
import numpy as np
from torch_geometric.data import Data


# === Function to convert a connectome matrix into a graph with multimodal node features ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    # === Build edge index and edge attributes from upper triangle
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)

    # === Get FA and Volume: shape [84, 2]
    fa_vol = node_features_dict[subject_id]  # [84, 2]

    # === Add MD as zeros: shape [84, 1]
    md_zeros = torch.zeros((84, 1), dtype=torch.float)

    # === Compute clustering coefficient per node: shape [84, 1]
    clustering_tensor = compute_nodewise_clustering_coefficients(matrix)

    # === Concatenate: [FA, MD=0, Volume, Clustering] → [84, 4]
    full_node_features = torch.cat([fa_vol[:, 0:1], md_zeros, fa_vol[:, 1:2], clustering_tensor], dim=1)

    # === Optional: scale to match AD-DECODE (if needed)
    node_features = 0.5 * full_node_features.to(device)

    return edge_index, edge_attr, node_features





#GRAPH METRICS

    
################## CLUSTERING COEFFICIENT ###############

def compute_clustering_coefficient(matrix):
    """
    Computes the average clustering coefficient of a graph represented by a matrix.

    Parameters:
    - matrix (pd.DataFrame): Connectivity matrix (84x84)

    Returns:
    - float: average clustering coefficient
    """
    G = nx.from_numpy_array(matrix.to_numpy())
    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]  # Add weights from matrix

    return nx.average_clustering(G, weight="weight")






################### PATH LENGTH ##################

def compute_path_length(matrix):
    """
    Computes the characteristic path length of the graph (average shortest path length).
    Converts weights to distances as 1 / weight.
    Uses the largest connected component if the graph is disconnected.
    
    Parameters:
    - matrix (pd.DataFrame): 84x84 connectome matrix
    
    Returns:
    - float: average shortest path length
    """
    # === 1. Create graph from matrix ===
    G = nx.from_numpy_array(matrix.to_numpy())

    # === 2. Assign weights and convert to distances ===
    for u, v, d in G.edges(data=True):
        weight = matrix.iloc[u, v]
        d["distance"] = 1.0 / weight if weight > 0 else float("inf")

    # === 3. Ensure graph is connected ===
    if not nx.is_connected(G):
        # Take the largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # === 4. Compute average shortest path length ===
    try:
        return nx.average_shortest_path_length(G, weight="distance")
    except:
        return float("nan")




# --- Assign computed metrics to metadata ---
df_matched_adni_healthy_withGM = df_matched_adni_healthy.reset_index(drop=True)
df_matched_adni_healthy_withGM["Clustering_Coeff"] = np.nan
df_matched_adni_healthy_withGM["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes_adni.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        df_matched_adni_healthy_withGM.loc[
            df_matched_adni_healthy_withGM["connectome_key"] == subject, "Clustering_Coeff"
        ] = clustering
        df_matched_adni_healthy_withGM.loc[
            df_matched_adni_healthy_withGM["connectome_key"] == subject, "Path_Length"
        ] = path
    except Exception as e:
        print(f"Failed to compute metrics for subject {subject}: {e}")






############ DEMOGRAPHIC FEATURES (NORM) ########################

from sklearn.preprocessing import LabelEncoder
import torch
from scipy.stats import zscore

print("=== PROCESSING ADNI DEMOGRAPHIC FEATURES ===")

# === Reset index ===
df_matched_adni_healthy_withGM = df_matched_adni_healthy_withGM.reset_index(drop=True)


# === Match genotype string format with AD-DECODE: e.g., "3_4" → "APOE34"
df_matched_adni_healthy_withGM["genotype"] = (
    "APOE" + df_matched_adni_healthy_withGM["APOE A1"].astype(str) + df_matched_adni_healthy_withGM["APOE A2"].astype(str)
)


# === Define selected feature groups ===
categorical_cols = ["Sex", "genotype"]

# === Drop rows with missing values ===
df_matched_adni_healthy_withGM = df_matched_adni_healthy_withGM.dropna(subset=categorical_cols).reset_index(drop=True)




# === Label encode Sex in ADNI (like in AD-DECODE) ===
le_sex = LabelEncoder()
df_matched_adni_healthy_withGM["sex_encoded"] = le_sex.fit_transform(df_matched_adni_healthy_withGM["Sex"].astype(str))


# === Label Encode Genotype ===
le = LabelEncoder()
df_matched_adni_healthy_withGM["genotype_encoded"] = le.fit_transform(df_matched_adni_healthy_withGM["genotype"].astype(str))





# --- Normalize numerical  columns ---
numerical_cols = [ "Clustering_Coeff", "Path_Length"]


df_matched_adni_healthy_withGM[numerical_cols] = df_matched_adni_healthy_withGM[numerical_cols].apply(zscore)




# ===============================
#  Build Metadata, graph metrics and PCA Tensors
# ===============================


# 1. Demographic tensor with zeros for missing systolic/diastolic
subject_to_demographic_tensor_adni = {
    row["connectome_key"]: torch.tensor([
        0.0, 0.0,                     # systolic, diastolic → zeros
        row["sex_encoded"],
        row["genotype_encoded"]
    ], dtype=torch.float)
    for _, row in df_matched_adni_healthy_withGM.iterrows()
}

# 2. Graph metric tensor remains the same
subject_to_graphmetric_tensor_adni = {
    row["connectome_key"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in df_matched_adni_healthy_withGM.iterrows()
}

# 3. PCA tensor → zeros
subject_to_pca_tensor_adni = {
    row["connectome_key"]: torch.zeros(10, dtype=torch.float)         
    for _, row in df_matched_adni_healthy_withGM.iterrows()
}





#####################  DEVICE CONFIGURATION  #######################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




#################  CONVERT MATRIX TO GRAPH  ################


from torch_geometric.data import Data

graph_data_list_adni = []
final_adni_subjects_with_all_data = []

for subject, matrix_log in log_thresholded_connectomes_adni.items():
    try:
        # === Skip if any required input is missing ===
        if subject not in subject_to_demographic_tensor_adni:
            continue
        if subject not in subject_to_graphmetric_tensor_adni:
            continue
        if subject not in subject_to_pca_tensor_adni:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # === Convert matrix to graph (node features: FA, Vol)
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log,
            device=torch.device("cpu"),
            subject_id=subject,
            node_features_dict=normalized_node_features_dict  # Ensure shape: [84, 2]
        )

        # === Get target age
        age_row = df_matched_adni_healthy_withGM.loc[
            df_matched_adni_healthy_withGM["connectome_key"] == subject, "Age"
        ]
        if age_row.empty:
            continue
        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        # === Concatenate global features (demo + graph + PCA) → [16]
        demo_tensor = subject_to_demographic_tensor_adni[subject]     # [4]
        graph_tensor = subject_to_graphmetric_tensor_adni[subject]    # [2]
        pca_tensor = subject_to_pca_tensor_adni[subject]              # [10]
        global_feat = torch.cat([demo_tensor, graph_tensor, pca_tensor], dim=0)  # [16]

        # === Create Data object
        data = Data(
            x=node_features,                      
            edge_index=edge_index,               # [2, num_edges]
            edge_attr=edge_attr,                 # [num_edges]
            y=age,                               # [1]
            global_features=global_feat.unsqueeze(0)  # [1, 16]
        )
        data.subject_id = subject

        # === Store graph
        graph_data_list_adni.append(data)
        final_adni_subjects_with_all_data.append(subject)

        # === Print one example
        if len(graph_data_list_adni) == 1:
            print("\nExample ADNI Data object:")
            print("→ Node features shape:", data.x.shape)           # [84, 4?]
            print("→ Edge index shape:", data.edge_index.shape)     # [2, ~3500]
            print("→ Edge attr shape:", data.edge_attr.shape)       # [~3500]
            print("→ Global features shape:", data.global_features.shape)  # [1, 16]
            print("→ Target age (y):", data.y.item())

    except Exception as e:
        print(f"Failed to process subject {subject}: {e}")

# === Save
torch.save(graph_data_list_adni, "graph_data_list_adni.pt")
print("Saved: graph_data_list_adni.pt")



#####################  DEVICE CONFIGURATION  #######################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")




######################  DEFINE MODEL  #########################

# MULTIHEAD-> one head for each global feature

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class BrainAgeGATv2(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()

        # === NODE FEATURES EMBEDDING ===
        # Each brain region (node) has 4 features: FA, MD, Volume, Clustering coefficient. (md zeros)
        # These are embedded into a higher-dimensional representation (64).
        self.node_embed = nn.Sequential(
            nn.Linear(4, 64),  # Project node features to 64-dimensional space
            nn.ReLU(),
            nn.Dropout(0.15)
            
        )

        # === GATv2 LAYERS WITH EDGE ATTRIBUTES ===
        # These layers use the connectome (edge weights) to propagate information.
        # edge_dim=1 means each edge has a scalar weight (from the functional connectome).
        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)  # Normalize output (16*8 = 128 channels)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)  # Regularization

        # === GLOBAL FEATURE BRANCHES ===
        # These process metadata that is not node-specific, grouped into 3 categories.

        # Demographic + physiological metadata (sex, systolic, diastolic, genotype)
        self.meta_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Graph-level metrics: global clustering coefficient and path length
        self.graph_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Top 10 PCA components from gene expression data, selected for age correlation
        self.pca_head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # === FINAL FUSION MLP ===
        # Combines graph-level information from GNN and global features
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 128),  # 128 from GNN output + 64 from metadata branches
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Final output: predicted brain age
        )

    def forward(self, data):
        # === GRAPH INPUTS ===
        x = data.x               # Node features: shape [num_nodes, 4]
        edge_index = data.edge_index  # Graph connectivity (edges)
        edge_attr = data.edge_attr    # Edge weights from functional connectome

        # === NODE EMBEDDING ===
        x = self.node_embed(x)  # Embed the node features

        # === GNN BLOCK 1 ===
        x = self.gnn1(x, edge_index, edge_attr=edge_attr)  # Attention using connectome weights
        x = self.bn1(x)
        x = F.relu(x)

        # === GNN BLOCK 2 with residual connection ===
        x_res1 = x  # Save for residual
        x = self.gnn2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        # === GNN BLOCK 3 with residual ===
        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        # === GNN BLOCK 4 with residual ===
        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr=edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        # === POOLING ===
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)  # Aggregate node embeddings into graph-level representation

        # === GLOBAL FEATURES ===
        # Shape: [batch_size, 1, 16] → remove extra dimension
        global_feats = data.global_features.to(x.device).squeeze(1)

        # Process each global feature group
        meta_embed = self.meta_head(global_feats[:, 0:4])    # Demographics
        graph_embed = self.graph_head(global_feats[:, 4:6])  # Clustering and path length
        pca_embed = self.pca_head(global_feats[:, 6:])       # Top 10 gene PCs

        # Concatenate all global embeddings
        global_embed = torch.cat([meta_embed, graph_embed, pca_embed], dim=1)  # Shape: [batch_size, 64]

        # === FUSION AND PREDICTION ===
        x = torch.cat([x, global_embed], dim=1)  # Combine GNN and metadata features
        x = self.fc(x)  # Final MLP to predict age

        return x  # Output: predicted age





    
     
    
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Usamos el DataLoader de torch_geometric

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)  # GPU
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # GPU
            output = model(data).view(-1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)








import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import numpy as np

# Training parameters
epochs = 300
patience = 40  # Early stopping

k =  7 # Folds
batch_size = 6

# === Initialize losses ===
all_train_losses = []
all_test_losses = []

all_early_stopping_epochs = []  




#Age bins 

# === Extract subject IDs from graph data
graph_subject_ids_adni = [data.subject_id for data in graph_data_list_adni]

# === Filter ADNI metadata to match graph subjects
df_filtered_adni = df_matched_adni_healthy_withGM[
    df_matched_adni_healthy_withGM["connectome_key"].isin(graph_subject_ids_adni)
].copy()

# Ensure order matches graphs
df_filtered_adni = df_filtered_adni.drop_duplicates(subset="connectome_key", keep="first")
df_filtered_adni = df_filtered_adni.set_index("connectome_key")
df_filtered_adni = df_filtered_adni.loc[graph_subject_ids_adni].reset_index()

# === Extract ages and compute bins
ages = df_filtered_adni["Age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)






# Stratified split by age bins
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


repeats_per_fold = 10  


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_adni, age_bins)):

    print(f'\n--- Fold {fold+1}/{k} ---')

    train_data = [graph_data_list_adni[i] for i in train_idx]
    test_data = [graph_data_list_adni[i] for i in test_idx]

    fold_train_losses = []
    fold_test_losses = []

    for repeat in range(repeats_per_fold):
        print(f'  > Repeat {repeat+1}/{repeats_per_fold}')
        
        early_stop_epoch = None  

        seed_everything(42 + repeat)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = BrainAgeGATv2(global_feat_dim=16).to(device)  

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        best_loss = float('inf')
        patience_counter = 0

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion)
            test_loss = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"model_adni1_fold_{fold+1}_rep_{repeat+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_epoch = epoch + 1  
                    print(f"    Early stopping triggered at epoch {early_stop_epoch}.")  
                    break


            scheduler.step()

        if early_stop_epoch is None:
                early_stop_epoch = epochs  
        all_early_stopping_epochs.append((fold + 1, repeat + 1, early_stop_epoch))


        fold_train_losses.append(train_losses)
        fold_test_losses.append(test_losses)

    all_train_losses.append(fold_train_losses)
    all_test_losses.append(fold_test_losses)







#################  LEARNING CURVE GRAPH (MULTIPLE REPEATS)  ################

plt.figure(figsize=(10, 6))

# Plot average learning curves across all repeats for each fold
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses[fold][rep], label=f'Train Loss - Fold {fold+1} Rep {rep+1}', linestyle='dashed', alpha=0.5)
        plt.plot(all_test_losses[fold][rep], label=f'Test Loss - Fold {fold+1} Rep {rep+1}', alpha=0.5)

plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curve (All Repeats)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)
plt.show()


# ==== LEARNING CURVE PLOT (MEAN ± STD) ====

import numpy as np
import matplotlib.pyplot as plt

# Compute mean and std for each epoch across all folds and repeats
avg_train = []
avg_test = []

for epoch in range(epochs):
    epoch_train = []
    epoch_test = []
    for fold in range(k):
        for rep in range(repeats_per_fold):
            if epoch < len(all_train_losses[fold][rep]):
                epoch_train.append(all_train_losses[fold][rep][epoch])
                epoch_test.append(all_test_losses[fold][rep][epoch])
    avg_train.append((np.mean(epoch_train), np.std(epoch_train)))
    avg_test.append((np.mean(epoch_test), np.std(epoch_test)))

# Unpack into arrays
train_mean, train_std = zip(*avg_train)
test_mean, test_std = zip(*avg_test)

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
plt.title("Learning Curve (Mean ± Std Across All Folds/Repeats)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





#####################  PREDICTION & METRIC ANALYSIS ACROSS FOLDS/REPEATS  #####################


from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === Initialize storage ===
fold_mae_list = []
fold_r2_list = []
all_y_true = []
all_y_pred = []


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_adni, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_data = [graph_data_list_adni[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = BrainAgeGATv2(global_feat_dim=16).to(device)  

        model.load_state_dict(torch.load(f"model_adni1_fold_{fold+1}_rep_{rep+1}.pt"))  # Load correct model
        model.eval()

        # === Load model if saved by repetition ===
        # model.load_state_dict(torch.load(f"model_fold_{fold+1}_rep_{rep+1}.pt"))

        y_true_repeat = []
        y_pred_repeat = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data).view(-1)
                y_pred_repeat.extend(output.cpu().tolist())
                y_true_repeat.extend(data.y.cpu().tolist())

        # Store values for this repeat
        mae = mean_absolute_error(y_true_repeat, y_pred_repeat)
        r2 = r2_score(y_true_repeat, y_pred_repeat)
        repeat_maes.append(mae)
        repeat_r2s.append(r2)

        all_y_true.extend(y_true_repeat)
        all_y_pred.extend(y_pred_repeat)

    fold_mae_list.append(repeat_maes)
    fold_r2_list.append(repeat_r2s)

    print(f">> Fold {fold+1} | MAE: {np.mean(repeat_maes):.2f} ± {np.std(repeat_maes):.2f} | R²: {np.mean(repeat_r2s):.2f} ± {np.std(repeat_r2s):.2f}")

# === Final aggregate results ===
all_maes = np.array(fold_mae_list).flatten()
all_r2s = np.array(fold_r2_list).flatten()

print("\n================== FINAL METRICS ==================")
print(f"Global MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f}")
print(f"Global R²:  {np.mean(all_r2s):.2f} ± {np.std(all_r2s):.2f}")
print("===================================================")






##############################################################
# PREDICTION & METRIC ANALYSIS — ADNI  (SAVE TO CSV)
###############################################################
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

# ---------------------------  CONFIG  ---------------------------
OUT_DIR = "ani1_training_eval_plots_save"        # where CSV + figs go
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE       = 6                                 # must match training
REPEATS_PER_FOLD = 10
N_FOLDS          = 7                                 # k en tu entrenamiento

# ---------------------------------------------------------------
# 1)  splits used in training
# ---------------------------------------------------------------
ages = np.array([data.y.item() for data in graph_data_list_adni]) 
age_bins = pd.qcut(ages, q=5, labels=False)                          

skf_adni = StratifiedKFold(
    n_splits=N_FOLDS, shuffle=True, random_state=42)

# ---------------------------------------------------------------
# 2) lists
# ---------------------------------------------------------------
fold_mae, fold_rmse, fold_r2       = [], [], []
all_y_true, all_y_pred             = [], []
all_subject_ids, fold_tags         = [], []
repeat_tags                        = []

# ---------------------------------------------------------------
# 3) Loop per fold × repeat
# ---------------------------------------------------------------
for fold, (train_idx, test_idx) in enumerate(skf_adni.split(
        graph_data_list_adni, age_bins)):

    print(f"\n--- Evaluating AD-DECODE Fold {fold+1}/{N_FOLDS} ---")
    test_loader = DataLoader(
        [graph_data_list_adni[i] for i in test_idx],
        batch_size=BATCH_SIZE, shuffle=False)

    mae_rep, rmse_rep, r2_rep = [], [], []            # métricas por repeat

    for rep in range(REPEATS_PER_FOLD):
        print(f"  > Repeat {rep+1}/{REPEATS_PER_FOLD}")

        # ----- Load trained model -----
        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        ckpt_path = f"model_adni1_fold_{fold+1}_rep_{rep+1}.pt"
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        # ----- Predictions -----
        y_true, y_pred, subj_ids = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch).view(-1)           # predicted age
                trues = batch.y.view(-1)                # real age

                y_pred.extend(preds.cpu().tolist())
                y_true.extend(trues.cpu().tolist())
                subj_ids.extend([str(s) for s in batch.subject_id])

        # ----- Metrics -----
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2   = r2_score(y_true, y_pred)

        mae_rep.append(mae)
        rmse_rep.append(rmse)
        r2_rep.append(r2)

        # ----- Save in lists -----
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_subject_ids.extend(subj_ids)
        fold_tags.extend([fold+1] * len(y_true))
        repeat_tags.extend([rep+1] * len(y_true))

    # ----- Summary per fold -----
    fold_mae.append(mae_rep)
    fold_rmse.append(rmse_rep)
    fold_r2.append(r2_rep)

    print(f">> Fold {fold+1} | "
          f"MAE:  {np.mean(mae_rep):.2f} ± {np.std(mae_rep):.2f} | "
          f"RMSE: {np.mean(rmse_rep):.2f} ± {np.std(rmse_rep):.2f} | "
          f"R²:   {np.mean(r2_rep):.2f} ± {np.std(r2_rep):.2f}")

# ---------------------------------------------------------------
# 4) Global metrics
# ---------------------------------------------------------------
all_mae  = np.concatenate(fold_mae)
all_rmse = np.concatenate(fold_rmse)
all_r2   = np.concatenate(fold_r2)

print("\n================== FINAL METRICS ADNI ==================")
print(f"Global MAE:  {all_mae.mean():.2f} ± {all_mae.std():.2f}")
print(f"Global RMSE: {all_rmse.mean():.2f} ± {all_rmse.std():.2f}")
print(f"Global R²:   {all_r2.mean():.2f} ± {all_r2.std():.2f}")
print("=============================================================\n")

# ---------------------------------------------------------------
# 5) Save CSV with all predictions
# ---------------------------------------------------------------
df_preds_adni = pd.DataFrame({
    "Subject_ID":    all_subject_ids,
    "Real_Age":      all_y_true,
    "Predicted_Age": all_y_pred,
    "Fold":          fold_tags,
    "Repeat":        repeat_tags
})

csv_path = os.path.join(OUT_DIR, "cv_predictions_adni1.csv")
df_preds_adni.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")




######################  PLOT TRUE VS PREDICTED AGES  ######################


plt.figure(figsize=(8, 6))

# Scatter plot of true vs predicted ages
plt.scatter(all_y_true, all_y_pred, alpha=0.7, edgecolors='k', label="Predictions")

# Calculate min/max values for axes with a small margin
min_val = min(min(all_y_true), min(all_y_pred))
max_val = max(max(all_y_true), max(all_y_pred))
margin = (max_val - min_val) * 0.05  # 5% margin for better spacing

# Plot the ideal diagonal line (y = x)
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="dashed", label="Ideal (y=x)")

# Set axis limits with margin for better visualization
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

# Metrics to display (Mean Absolute Error and R-squared)
textstr = f"MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f}\nR²: {np.mean(all_r2s):.2f} ± {np.std(all_r2s):.2f}"
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

# Axis labels and title
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Ages (All Repeats)")

# Legend and grid
plt.legend(loc="upper left")
plt.grid(True)

# No need for equal scaling here, as it compresses the data visually
# plt.axis("equal")

plt.show()





######## TRAINING WITH ALL DATA FOR PRETRAINING #############



print("\n=== Training Final Model on All ADNI Subjects ===")

from torch_geometric.loader import DataLoader

# Create DataLoader with all ADNI data
final_train_loader = DataLoader(graph_data_list_adni, batch_size=6, shuffle=True)

# Initialize model
final_model = BrainAgeGATv2(global_feat_dim=16).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Loss function
criterion = torch.nn.SmoothL1Loss(beta=1)

# Fixed number of epochs (based on early stopping analysis)
epochs = 120

# Training loop
for epoch in range(epochs):
    final_model.train()
    total_loss = 0
    for data in final_train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = final_model(data).view(-1)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(final_train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    scheduler.step()

# Save model after full training
torch.save(final_model.state_dict(), "brainage_adni1_pretrained.pt")
print("\nFinal ADNI model saved as 'brainage_adni1_pretrained.pt'")





