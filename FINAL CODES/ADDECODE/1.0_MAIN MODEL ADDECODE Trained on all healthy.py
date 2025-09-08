###################################################################
# ADDECODE - Preprocessing, Training and evaluation on all healthy
###################################################################

# Preprocess connectomes , metadata, node features
# Convert each subject into a pytorch graph object 
# Trains a GATv2 model asiing 7 fold stratified CV - 10 repeats 
# Evaluate model performance and save the model to predict on all risks



#Preprocess connectomes -> Log(x+1) and Threshold

# Metadata:  -> sex,genotype,systolc, diasstolic, clustering coeff, path lentgh 10 PCA (zscores)
    # Zscore normalizing global features (metadata, graph metrics, pcas)
    # Using the top 10 most age-correlated PCs, according to SPEARMAN PC trait Correlation (PCA enrichment)
    # top 10 zscored
    # Using less metadata features(sex,genotype,systolc, diasstolic)​
    # Sex label encoded
    # (sex: Label encoded and one hot encoded (similar)​ )
    # Using only clustering coeff and path lentgh as graph metrics​
    # MULTI HEAD 1 for each group( metadata, graph metrics, pcas)

#Node features -> FA,MD,Volume, clustering coefficient
    # ADDED CLUSTERING COEF AS  A NODE FEATURE​

#MODEL
    #4 GATv2 layers​
    # Residual connections between 1 and 2,2 and 3, 3 and 4​
    # Batch norm​
    # Concat true, heads 8​
    # Patience 40 ​
    
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





#Remove AD and MCI

# === Print risk distribution if available ===
if "Risk" in df_matched_connectomes.columns:
    risk_filled = df_matched_connectomes["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()



print("FILTERING OUT AD AND MCI SUBJECTS")

# === Keep only healthy control subjects ===
df_matched_addecode_healthy = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
].copy()

print(f"Subjects before removing AD/MCI: {len(df_matched_connectomes)}")
print(f"Subjects after removing AD/MCI: {len(df_matched_addecode_healthy)}")
print()


# === Show updated 'Risk' distribution ===
if "Risk" in df_matched_addecode_healthy.columns:
    risk_filled = df_matched_addecode_healthy["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()



#Connectomes
# === Filter connectomes to include only those from non-AD/MCI subjects ===
matched_connectomes_healthy_addecode = {
    row["MRI_Exam_fixed"]: matched_connectomes[row["MRI_Exam_fixed"]]
    for _, row in df_matched_addecode_healthy.iterrows()
}

# === Confirmation of subject count
print(f"Connectomes selected (excluding AD/MCI): {len(matched_connectomes_healthy_addecode)}")
print()


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

print(df_matched_addecode_healthy.head())



# Fix id formats

# === Fix ID format in PCA DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE_1' → 'ADDECODE1'
df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)



# === Fix Subject format in metadata DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE1' → 'ADDECODE1'
df_matched_addecode_healthy["IDRNA_fixed"] = df_matched_addecode_healthy["IDRNA"].str.upper().str.replace("_", "", regex=False)




###### MATCH PCA GENES WITH METADATA############

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_healthy_withConnectome = df_matched_addecode_healthy.merge(df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed")


#Numbers

# === Show how many healthy subjects with PCA and connectome you have
print(f" Healthy subjects with metadata connectome: {df_matched_addecode_healthy.shape[0]}")
print()

print(f" Healthy subjects with metadata PCA & connectome: {df_metadata_PCA_healthy_withConnectome.shape[0]}")
print()


# Get the full set of subject IDs (DWI_fixed) in healthy set
all_healthy_ids = set(df_matched_addecode_healthy["MRI_Exam_fixed"])

# Get the subject IDs (DWI_fixed) that matched with PCA
healthy_with_pca_ids = set(df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"])

# Compute the difference: healthy subjects without PCA
healthy_without_pca_ids = all_healthy_ids - healthy_with_pca_ids

# Filter the original healthy metadata for those subjects
df_healthy_without_pca = df_matched_addecode_healthy[
    df_matched_addecode_healthy["MRI_Exam_fixed"].isin(healthy_without_pca_ids)
]


# Print result
print(f" Healthy subjects with connectome but NO PCA: {df_healthy_without_pca.shape[0]}")
print()

print(df_healthy_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]])






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

import re

# Clean and deduplicate FA subjects based on numeric ID (e.g. "02842")
cleaned_fa = {}

for subj in df_fa_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_fa:
            cleaned_fa[subj_id] = df_fa_transposed.loc[subj]

# Convert cleaned data to DataFrame
df_fa_transposed_cleaned = pd.DataFrame.from_dict(cleaned_fa, orient="index")
df_fa_transposed_cleaned.index.name = "subject_id"



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


# Clean and deduplicate MD subjects based on numeric ID (e.g. "02842")
cleaned_md = {}

for subj in df_md_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_md:
            cleaned_md[subj_id] = df_md_transposed.loc[subj]

df_md_transposed_cleaned = pd.DataFrame.from_dict(cleaned_md, orient="index")
df_md_transposed_cleaned.index.name = "subject_id"




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

# Clean and deduplicate Volume subjects based on numeric ID (e.g. "02842")
cleaned_vol = {}

for subj in df_vol_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        if subj_id not in cleaned_vol:
            cleaned_vol[subj_id] = df_vol_transposed.loc[subj]

df_vol_transposed_cleaned = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_transposed_cleaned.index.name = "subject_id"


# === Combine FA + MD + Vol per subject using cleaned DataFrames ===

multimodal_features_dict = {}

# Use subject IDs from FA as reference (already cleaned to 5-digit keys)
for subj_id in df_fa_transposed_cleaned.index:
    # Check that this subject also exists in MD and Vol
    if subj_id in df_md_transposed_cleaned.index and subj_id in df_vol_transposed_cleaned.index:
        fa = torch.tensor(df_fa_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        md = torch.tensor(df_md_transposed_cleaned.loc[subj_id].values, dtype=torch.float)
        vol = torch.tensor(df_vol_transposed_cleaned.loc[subj_id].values, dtype=torch.float)

        # Stack the 3 modalities: [84 nodes, 3 features (FA, MD, Vol)]
        stacked = torch.stack([fa, md, vol], dim=1)

        # Store with subject ID as key
        multimodal_features_dict[subj_id] = stacked



print()
print(" Subjects with FA, MD, and Vol features:", len(multimodal_features_dict))

fa_md_vol_ids = set(multimodal_features_dict.keys())
pca_ids = set(df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"])
connectome_ids = set(matched_connectomes_healthy_addecode.keys())

final_overlap = fa_md_vol_ids & pca_ids & connectome_ids

print(" Subjects with FA/MD/Vol + PCA + Connectome:", len(final_overlap))

# Sample one subject from the dictionary
example_id = list(multimodal_features_dict.keys())[25]
print(" Example subject ID:", example_id)

# Check that this subject also exists in metadata and connectomes
in_metadata = example_id in df_metadata_PCA_healthy_withConnectome["MRI_Exam_fixed"].values
in_connectome = example_id in matched_connectomes_healthy_addecode

print(f" In metadata: {in_metadata}")
print(f" In connectomes: {in_connectome}")

# Print first few FA/MD/Vol values before normalization
example_tensor = multimodal_features_dict[example_id]
print(" First 5 nodes (FA):", example_tensor[:5, 0])
print(" First 5 nodes (MD):", example_tensor[:5, 1])
print(" First 5 nodes (Vol):", example_tensor[:5, 2])
print()





# === Normalization node-wise  ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # [N_subjects, 84, 3]
    means = all_features.mean(dim=0)  # [84, 3]
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Normalization
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
for subject, matrix in matched_connectomes_healthy_addecode.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=70)
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
addecode_healthy_metadata_pca = df_metadata_PCA_healthy_withConnectome.reset_index(drop=True)
addecode_healthy_metadata_pca["Clustering_Coeff"] = np.nan
addecode_healthy_metadata_pca["Path_Length"] = np.nan

for subject, matrix_log in log_thresholded_connectomes.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "Clustering_Coeff"
        ] = clustering
        addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "Path_Length"
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
addecode_healthy_metadata_pca["sex_encoded"] = le_sex.fit_transform(addecode_healthy_metadata_pca["sex"].astype(str))


# --- Label encode genotype ---
le = LabelEncoder()
addecode_healthy_metadata_pca["genotype"] = le.fit_transform(addecode_healthy_metadata_pca["genotype"].astype(str))

# --- Normalize numerical and PCA columns ---
numerical_cols = ["Systolic", "Diastolic", "Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3'] #Top 10 from SPEARMAN  corr (enrich)

addecode_healthy_metadata_pca[numerical_cols] = addecode_healthy_metadata_pca[numerical_cols].apply(zscore)
addecode_healthy_metadata_pca[pca_cols] = addecode_healthy_metadata_pca[pca_cols].apply(zscore)



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
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

# === 2. Graph metric tensor (clustering coefficient, path length) ===
subject_to_graphmetric_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"]
    ], dtype=torch.float)
    for _, row in addecode_healthy_metadata_pca.iterrows()
}

# === 3. PCA tensor (top 10 age-correlated components) ===
subject_to_pca_tensor = {
    row["MRI_Exam_fixed"]: torch.tensor(row[pca_cols].values.astype(np.float32))
    for _, row in addecode_healthy_metadata_pca.iterrows()
}




#################  CONVERT MATRIX TO GRAPH  ################

graph_data_list_addecode = []
final_subjects_with_all_data = []  # Para verificar qué sujetos sí se procesan

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
        age_row = addecode_healthy_metadata_pca.loc[
            addecode_healthy_metadata_pca["MRI_Exam_fixed"] == subject, "age"
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
            print("→ Node features shape:", data.x.shape)           # Ecpected: [84, 4]
            print("→ Edge index shape:", data.edge_index.shape)     # Ecpected: [2, ~3500]
            print("→ Edge attr shape:", data.edge_attr.shape)       # Ecpected: [~3500]
            print("→ Global features shape:", data.global_features.shape)  # Ecpected: [1, 16]
            print("→ Target age (y):", data.y.item())


    except Exception as e:
        print(f" Failed to process subject {subject}: {e}")

# === Save processed graph data for reuse
import torch
torch.save(graph_data_list_addecode, "graph_data_list_addecode.pt")
print("Saved: graph_data_list_addecode.pt")


# Check

# === Report stats ===
print()
expected = set(subject_to_pca_tensor.keys())
actual = set(final_subjects_with_all_data)
missing = expected - actual

print(f" Subjects with PCA but no graph: {missing}")
print(f" Total graphs created: {len(actual)} / Expected: {len(expected)}")



print()


example_subject = list(subject_to_pca_tensor.keys())[0]
print("Demo:", subject_to_demographic_tensor[example_subject].shape)
print("Graph:", subject_to_graphmetric_tensor[example_subject].shape)
print("PCA:", subject_to_pca_tensor[example_subject].shape)
print("Global:", torch.cat([
    subject_to_demographic_tensor[example_subject],
    subject_to_graphmetric_tensor[example_subject],
    subject_to_pca_tensor[example_subject]
], dim=0).shape)

print()




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
        # Each brain region (node) has 4 features: FA, MD, Volume, Clustering coefficient.
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
graph_subject_ids = [data.subject_id for data in graph_data_list_addecode]

# === Filter and sort metadata to match only graph subjects
df_filtered = addecode_healthy_metadata_pca[
    addecode_healthy_metadata_pca["MRI_Exam_fixed"].isin(graph_subject_ids)
].copy()

# Double-check: remove any unexpected mismatches
df_filtered = df_filtered.drop_duplicates(subset="MRI_Exam_fixed", keep="first")
df_filtered = df_filtered.set_index("MRI_Exam_fixed")
df_filtered = df_filtered.loc[df_filtered.index.intersection(graph_subject_ids)]
df_filtered = df_filtered.loc[graph_subject_ids].reset_index()

# Final check
print(" Final matched lengths:")
print("  len(graphs):", len(graph_data_list_addecode))
print("  len(metadata):", len(df_filtered))

# === Extract final age vector and compute age bins
ages = df_filtered["age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)

print(" Aligned bins:", len(age_bins))








# Stratified split by age bins
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


repeats_per_fold = 10  


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):

    print(f'\n--- Fold {fold+1}/{k} ---')

    train_data = [graph_data_list_addecode[i] for i in train_idx]
    test_data = [graph_data_list_addecode[i] for i in test_idx]

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
                torch.save(model.state_dict(), f"model_fold_{fold+1}_rep_{repeat+1}.pt")
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


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_data = [graph_data_list_addecode[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = BrainAgeGATv2(global_feat_dim=16).to(device)  

        model.load_state_dict(torch.load(f"model_fold_{fold+1}_rep_{rep+1}.pt"))  # Load correct model
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



###############################################################
# PREDICTION & METRIC ANALYSIS — AD-DECODE  (SAVE TO CSV)
###############################################################
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

# ---------------------------  CONFIG  ---------------------------
OUT_DIR = "addecode_training_eval_plots_save"        # where CSV + figs go
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE       = 6                                 # must match training
REPEATS_PER_FOLD = 10
N_FOLDS          = 7                                 # k en tu entrenamiento

# ---------------------------------------------------------------
# 1)  splits used in training
# ---------------------------------------------------------------
ages = np.array([data.y.item() for data in graph_data_list_addecode])  # 71 elementos
age_bins = pd.qcut(ages, q=5, labels=False)                            # 71 elementos

skf_addecode = StratifiedKFold(
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
for fold, (train_idx, test_idx) in enumerate(skf_addecode.split(
        graph_data_list_addecode, age_bins)):

    print(f"\n--- Evaluating AD-DECODE Fold {fold+1}/{N_FOLDS} ---")
    test_loader = DataLoader(
        [graph_data_list_addecode[i] for i in test_idx],
        batch_size=BATCH_SIZE, shuffle=False)

    mae_rep, rmse_rep, r2_rep = [], [], []            # métricas por repeat

    for rep in range(REPEATS_PER_FOLD):
        print(f"  > Repeat {rep+1}/{REPEATS_PER_FOLD}")

        # ----- Load trained model -----
        model = BrainAgeGATv2(global_feat_dim=16).to(device)
        ckpt_path = f"model_fold_{fold+1}_rep_{rep+1}.pt"
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

print("\n================== FINAL METRICS AD-DECODE ==================")
print(f"Global MAE:  {all_mae.mean():.2f} ± {all_mae.std():.2f}")
print(f"Global RMSE: {all_rmse.mean():.2f} ± {all_rmse.std():.2f}")
print(f"Global R²:   {all_r2.mean():.2f} ± {all_r2.std():.2f}")
print("=============================================================\n")

# ---------------------------------------------------------------
# 5) Save CSV with all predictions
# ---------------------------------------------------------------
df_preds_addecode = pd.DataFrame({
    "Subject_ID":    all_subject_ids,
    "Real_Age":      all_y_true,
    "Predicted_Age": all_y_pred,
    "Fold":          fold_tags,
    "Repeat":        repeat_tags
})

csv_path = os.path.join(OUT_DIR, "cv_predictions_addecode.csv")
df_preds_addecode.to_csv(csv_path, index=False)
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




#Evaluation is complete



#Now we are going to train a model on all healthy subjects, 
#and we save it to use it on the MCI and AD (excluded) and this healthy (it is okay because we already validated)






######################################
# FINAL MODEL TRAINING ON ALL HEALTHY
######################################

print("\n=== Training Final Model on All Healthy Subjects ===")

from torch_geometric.loader import DataLoader

# Create DataLoader with all healthy data
final_train_loader = DataLoader(graph_data_list_addecode, batch_size=6, shuffle=True)

# Initialize model
final_model = BrainAgeGATv2(global_feat_dim=16).to(device)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Loss function
criterion = torch.nn.SmoothL1Loss(beta=1)

# Fixed number of epochs (no early stopping)
epochs = 100

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
torch.save(final_model.state_dict(), "model_trained_on_all_healthy.pt")
print("\nFinal model saved as 'model_trained_on_all_healthy.pt'")



#We selected 100 epochs for the final model training based on the early stopping results observed during cross-validation. 
#Most repetitions across folds stopped between 45 and 80 epochs, with a few extending beyond 100




