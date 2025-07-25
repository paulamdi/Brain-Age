# Edges
#th70

# ADDECODE 

    #K fold cross validation
    #And then after already evaluating we train on all healthy subjects and save the model, to then predict on all risks on next script

    # MULTI HEAD 1 for each group( metadata, graph metrics, pcas)
    # Zscore normalizing global features (metadata, graph metrics, pcas)
    # Using the top 10 most age-correlated PCs, according to SPEARMAN PC trait Correlation (PCA enrichment)
    # top 10 zscored
    # Sex label encoded
   


    #4 gnn layers​
    #Residual connections between 1 and 2,2 and 3, 3 and 4​
    #Batch norm​
    #Concat true, heads 8​
    #Patience 40 ​
    #Using less metadata features(sex,genotype,systolc, diasstolic)​
    #sex: Label encoded and one hot encoded (similar)​
    #Using only clustering coeff and path lentgh as graph metrics​
    #ADDED CLUSTERING COEF AS  A NODE FEATURE​
    
    
import os

output_dir = "Model_with_shap_embeddings"
os.makedirs(output_dir, exist_ok=True)
    
    
    
    
    
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





# SHAP EMBEDDINGS

import pandas as pd

# Load the embeddings CSV
df_embed = pd.read_csv("/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/7_ SHAP_CL/shap_embeddings.csv")

# Add a new column with zero-padded IDs (e.g., 2110 → "02110")
df_embed["Subject_ID_fixed"] = df_embed["Subject_ID"].astype(str).str.zfill(5)


# Create the dictionary with fixed Subject_ID as key
embed_dict = {
    row["Subject_ID_fixed"]: torch.tensor(
        row.drop(labels=["Subject_ID", "Subject_ID_fixed"]).values.astype(np.float32)
    )
    for _, row in df_embed.iterrows()
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
        if subject not in embed_dict:
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


        
        # Add SHAP embedding (already verified to exist)
        shap_embedding = embed_dict[subject].unsqueeze(0)  # Shape: [1, 32]
        data.shap_embedding = shap_embedding





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
    def __init__(self):
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


        # embed
        self.mlp_shap = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )



        # === FINAL FUSION MLP ===
        # Combines graph-level information from GNN and global features
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 32 , 128),  # 128 from GNN output + 64 from metadata branches, 32 from shap embed
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

        x_shap = self.mlp_shap(data.shap_embedding)

        # Concatenate all global embeddings
        global_embed = torch.cat([meta_embed, graph_embed, pca_embed, x_shap], dim=1)  # Shape: [batch_size, 64]

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

        model = BrainAgeGATv2().to(device)  

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
#plt.show()

plt.savefig(os.path.join(output_dir, "ADDECODE_CLmodel_learningCurve_Allfolds.png"), dpi=300)
plt.close()



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
#plt.show()

plt.savefig(os.path.join(output_dir, "ADDECODE_CLmodel_learningCurve_MEAN.png"), dpi=300)
plt.close()



#####################  PREDICTION & METRIC ANALYSIS ACROSS FOLDS/REPEATS  #####################
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# === Storage ===
fold_mae_list, fold_rmse_list, fold_r2_list = [], [], []
all_y_true, all_y_pred, all_subject_ids = [], [], []   # ← añadimos IDs

for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_loader = DataLoader([graph_data_list_addecode[i] for i in test_idx],
                              batch_size=batch_size, shuffle=False)

    repeat_mae, repeat_rmse, repeat_r2 = [], [], []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = BrainAgeGATv2().to(device)
        model.load_state_dict(torch.load(f"model_fold_{fold+1}_rep_{rep+1}.pt"))
        model.eval()

        y_true_rep, y_pred_rep = [], []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)

                # --- Predictions & true ages ---
                preds = model(data).view(-1).cpu().tolist()
                trues = data.y.cpu().tolist()

                # --- Subject IDs (ensure string) 
                ids = [str(sid) for sid in data.subject_id]

                # --- Sanity check 
                assert len(preds) == len(trues) == len(ids), "Batch size mismatch!"

                # --- Extend master lists
                all_y_pred.extend(preds)
                all_y_true.extend(trues)
                all_subject_ids.extend(ids)

                # --- Extend repetition lists (for per-repeat metrics) ---
                y_pred_rep.extend(preds)
                y_true_rep.extend(trues)

        # === Metrics for this repeat ===
        mae  = mean_absolute_error(y_true_rep, y_pred_rep)
        rmse = np.sqrt(mean_squared_error(y_true_rep, y_pred_rep))
        r2   = r2_score(y_true_rep,  y_pred_rep)

        repeat_mae.append(mae)
        repeat_rmse.append(rmse)
        repeat_r2.append(r2)

    # === Store per-fold ===
    fold_mae_list.append(repeat_mae)
    fold_rmse_list.append(repeat_rmse)
    fold_r2_list.append(repeat_r2)

    print(f">> Fold {fold+1} | "
          f"MAE:  {np.mean(repeat_mae):.2f} ± {np.std(repeat_mae):.2f} | "
          f"RMSE: {np.mean(repeat_rmse):.2f} ± {np.std(repeat_rmse):.2f} | "
          f"R²:   {np.mean(repeat_r2):.2f} ± {np.std(repeat_r2):.2f}")

    



# === Global aggregates ===
all_maes  = np.concatenate(fold_mae_list)
all_rmses = np.concatenate(fold_rmse_list)
all_r2s   = np.concatenate(fold_r2_list)

print("\n================== FINAL METRICS ==================")
print(f"Global MAE:  {all_maes.mean():.2f} ± {all_maes.std():.2f}")
print(f"Global RMSE: {all_rmses.mean():.2f} ± {all_rmses.std():.2f}")
print(f"Global R²:   {all_r2s.mean():.2f} ± {all_r2s.std():.2f}")
print("===================================================")




#SCATTER PLOTS



######################  PLOT TRUE VS PREDICTED AGES  ######################

plt.figure(figsize=(8, 6))

# === Scatter plot of all predictions ===
plt.scatter(all_y_true, all_y_pred, alpha=0.7, edgecolors='k', label="Predictions")

# === Axis limits with margin ===
min_val = min(min(all_y_true), min(all_y_pred))
max_val = max(max(all_y_true), max(all_y_pred))
margin = (max_val - min_val) * 0.05

# === Ideal y = x line ===
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y = x)")

# === Linear trend line ===
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(np.array(all_y_true).reshape(-1, 1),
                             np.array(all_y_pred))
x_vals = np.array([min_val, max_val]).reshape(-1, 1)
y_vals = reg.predict(x_vals)
plt.plot(x_vals, y_vals, color="blue", label=f"Trend: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}")

# === Set axis limits ===
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)



textstr = (
    f"MAE:  {all_maes.mean():.2f} ± {all_maes.std():.2f}\n"
    f"RMSE: {all_rmses.mean():.2f} ± {all_rmses.std():.2f}\n"
    f"R²:   {all_r2s.mean():.2f} ± {all_r2s.std():.2f}"
)


plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

# === Labels and title ===
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Ages (All Repeats)")

# === Final touches ===
plt.legend(loc="upper left")
plt.grid(True)

# === Save plot ===
plt.savefig(os.path.join(output_dir, "ADDECODE_CLmodel_ScatterPlot.png"), dpi=300)
plt.close()






# =============================================================
# Scatter plot of ALL predictions (every repetition)
# with a nicer trend line format
# =============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

plt.figure(figsize=(8, 6))

# --- 1) Scatter of every prediction ---------------------------
plt.scatter(all_y_true, all_y_pred,
            alpha=0.7, edgecolors="k", label="Predictions")

# --- 2) Axis limits with 5 % margin ---------------------------
min_val = min(min(all_y_true), min(all_y_pred))
max_val = max(max(all_y_true), max(all_y_pred))
margin  = (max_val - min_val) * 0.05

# --- 3) Ideal diagonal (y = x) --------------------------------
plt.plot([min_val, max_val], [min_val, max_val],
         ls="--", color="red", label="Ideal (y = x)")

# --- 4) Linear trend line (lighter & full-range) ---------------
reg = LinearRegression().fit(
    np.array(all_y_true).reshape(-1, 1),
    np.array(all_y_pred)
)
trend_x = np.array([min_val, max_val]).reshape(-1, 1)  # endpoints
trend_y = reg.predict(trend_x)

slope     = reg.coef_[0]
intercept = reg.intercept_

plt.plot(trend_x, trend_y,
         color="blue", alpha=0.4, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.2f}")

# --- 5) Axis limits -------------------------------------------
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

# --- 6) Metrics text box (mean ± std) -------------------------
textstr = (
    f"MAE:  {all_maes.mean():.2f} ± {all_maes.std():.2f}\n"
    f"RMSE: {all_rmses.mean():.2f} ± {all_rmses.std():.2f}\n"
    f"R²:   {all_r2s.mean():.2f} ± {all_r2s.std():.2f}"
)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3",
                   edgecolor="black", facecolor="lightgray"))

# --- 7) Labels, title & legend --------------------------------
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (all repetitions)")
plt.title("Predicted vs Real Ages (All Repeats)")
plt.legend(loc="upper left")
plt.grid(True)

# --- 8) Save figure -------------------------------------------
plt.savefig(os.path.join(output_dir,
                         "ADDECODE_CLmodel_ScatterPlot_withTrend.png"),
            dpi=300)
plt.close()






# ------------------------------------------------------------------------------------------------
# 3) SCATTER 1 POINT PER SUBJECT (no trend line))
# ------------------------------------------------------------------------------------------------

# ============================================================
# 1) Compute one mean prediction per subject
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os

# --- Build a DataFrame with every repetition ----------------
df_preds = pd.DataFrame({
    "Subject_ID": all_subject_ids,     # string IDs
    "Real_Age":  all_y_true,           # ground-truth age
    "Predicted_Age": all_y_pred        # model output
})

# --- Average predictions per subject ------------------------
df_means = (df_preds
            .groupby("Subject_ID")
            .agg({"Real_Age": "first",           # same for every rep
                  "Predicted_Age": "mean"})      # mean across reps
            .reset_index())

# ============================================================
# 2) Metrics on subject-level means
# ============================================================
mae_mean  = mean_absolute_error(df_means["Real_Age"], df_means["Predicted_Age"])
rmse_mean = mean_squared_error(df_means["Real_Age"], df_means["Predicted_Age"], squared=False)
r2_mean   = r2_score(df_means["Real_Age"], df_means["Predicted_Age"])

# ============================================================
# 3) Scatter plot (one dot per subject)
# ============================================================
plt.figure(figsize=(8, 6))

# --- Scatter ------------------------------------------------
plt.scatter(df_means["Real_Age"], df_means["Predicted_Age"],
            alpha=0.8, edgecolors='k', label="Mean per Subject")

# --- Ideal diagonal ----------------------------------------
min_age = min(df_means["Real_Age"].min(), df_means["Predicted_Age"].min())
max_age = max(df_means["Real_Age"].max(), df_means["Predicted_Age"].max())
margin  = (max_age - min_age) * 0.05

plt.plot([min_age, max_age], [min_age, max_age],
         ls="--", color="red", label="Ideal (y = x)")

# --- Trend line --------------------------------------------
reg_m = LinearRegression().fit(
    df_means["Real_Age"].values.reshape(-1, 1),
    df_means["Predicted_Age"].values
)
plt.plot([min_age, max_age],
         reg_m.predict(np.array([min_age, max_age]).reshape(-1, 1)),
         color="blue", alpha=0.6, linewidth=2,
         label=f"Trend: y = {reg_m.coef_[0]:.2f}x + {reg_m.intercept_:.2f}")

# --- Axes limits -------------------------------------------
plt.xlim(min_age - margin, max_age + margin)
plt.ylim(min_age - margin, max_age + margin)

# --- Metrics box -------------------------------------------
textstr = (f"MAE:  {mae_mean:.2f}\n"
           f"RMSE: {rmse_mean:.2f}\n"
           f"R²:   {r2_mean:.2f}")
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3",
                   edgecolor="black", facecolor="lightgray"))

# --- Labels & styling --------------------------------------
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean of 10 reps)")
plt.title("Predicted vs Real Ages (Mean per Subject)")
plt.legend(loc="upper left")
plt.grid(True)

# --- Save figure -------------------------------------------
plt.savefig(os.path.join(output_dir, "ADDECODE_CLmodel_ScatterPlot_MEAN.png"),
            dpi=300)
plt.close()







# ------------------------------------------------------------------------------------------------
# 4) SCATTER ONE POINT PER SUBJECT + TREND 
# ------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# === Agrupar predicciones por sujeto y calcular la media ===
df_preds = pd.DataFrame({
    "Subject_ID": all_subject_ids,
    "Real_Age": all_y_true,
    "Predicted_Age": all_y_pred
})

df_means = (df_preds
            .groupby("Subject_ID")
            .agg({"Real_Age": "first", "Predicted_Age": "mean"})
            .reset_index())

# === Calcular métricas usando la media por sujeto ===
mae_mean  = mean_absolute_error(df_means["Real_Age"], df_means["Predicted_Age"])
rmse_mean = mean_squared_error(df_means["Real_Age"], df_means["Predicted_Age"], squared=False)
r2_mean   = r2_score(df_means["Real_Age"], df_means["Predicted_Age"])

# === Gráfico ===
plt.figure(figsize=(8, 6))

# --- Scatter plot ---
plt.scatter(df_means["Real_Age"], df_means["Predicted_Age"],
            alpha=0.8, edgecolors="k", label="Mean Prediction")

# --- Límites con margen ---
min_age = min(df_means["Real_Age"].min(), df_means["Predicted_Age"].min())
max_age = max(df_means["Real_Age"].max(), df_means["Predicted_Age"].max())
margin  = (max_age - min_age) * 0.05

# --- Línea ideal ---
plt.plot([min_age, max_age], [min_age, max_age],
         ls="--", color="red", label="Ideal (y = x)")

# --- Trend line con ecuación en leyenda ---
reg = LinearRegression().fit(df_means["Real_Age"].values.reshape(-1, 1),
                             df_means["Predicted_Age"].values)
trend_x = np.array([min_age, max_age]).reshape(-1, 1)
trend_y = reg.predict(trend_x)
slope = reg.coef_[0]
intercept = reg.intercept_

plt.plot(trend_x, trend_y,
         color="blue", alpha=0.4, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.2f}")

# --- Límites de ejes ---
plt.xlim(min_age - margin, max_age + margin)
plt.ylim(min_age - margin, max_age + margin)

# --- Cuadro de métricas (sin desviación) ---
textstr = (
    f"MAE:  {mae_mean:.2f}\n"
    f"RMSE: {rmse_mean:.2f}\n"
    f"R²:   {r2_mean:.2f}"
)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

# --- Estilo general ---
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean of 10 reps)")
plt.title("Predicted vs Real Ages (Mean per Subject)")
plt.legend(loc="upper left")
plt.grid(True)

# --- Guardar figura ---
plt.savefig(os.path.join(output_dir, "ADDECODE_CLmodel_ScatterPlot_MEAN_withTrendLegend.png"),
            dpi=300)
plt.close()





##########################################
# SCATTER PER SUBJECT ±1 STD (No Trend Line)
##########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# === 1. Aggregate mean & std per subject ==============================
df_stats = (df_preds
            .groupby("Subject_ID")
            .agg(Real_Age   = ('Real_Age',      'first'),
                 Pred_Mean  = ('Predicted_Age', 'mean'),
                 Pred_Std   = ('Predicted_Age', 'std'))
            .reset_index())

# === 2. Figure setup =================================================
plt.figure(figsize=(8, 6))

# Scatter with vertical error bars (±1 STD)
plt.errorbar(df_stats["Real_Age"],
             df_stats["Pred_Mean"],
             yerr=df_stats["Pred_Std"],
             fmt='o',
             ecolor='gray',
             elinewidth=1,
             capsize=3,
             alpha=0.85,
             markeredgecolor='k',
             label="Mean ± 1 STD")

# Ideal diagonal y = x
min_age = min(df_stats["Real_Age"].min(), df_stats["Pred_Mean"].min())
max_age = max(df_stats["Real_Age"].max(), df_stats["Pred_Mean"].max())
margin  = (max_age - min_age) * 0.05
plt.plot([min_age, max_age], [min_age, max_age],
         ls='--', color='red', linewidth=1.0, label='Ideal (y = x)')

# === 3. Styling =======================================================
plt.xlim(min_age - margin, max_age + margin)
plt.ylim(min_age - margin, max_age + margin)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean ± std)")
plt.title("Predicted vs Real Ages — per subject")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

# === 4. Global metrics box ===========================================
repeats = 10  # adjust if needed
maes, rmses, r2s = [], [], []

for rep in range(repeats):
    df_rep = df_preds.iloc[rep::repeats]          # rows for repetition `rep`
    df_rep_mean = (df_rep.groupby("Subject_ID")
                           .agg({"Real_Age": "first",
                                 "Predicted_Age": "mean"}))

    maes.append(mean_absolute_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"]))
    rmses.append(mean_squared_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"], squared=False))
    r2s.append(r2_score(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"]))

mae_mean, mae_std   = np.mean(maes),  np.std(maes)
rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
r2_mean,   r2_std   = np.mean(r2s),   np.std(r2s)

textstr = (f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}\n"
           f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
           f"R²:   {r2_mean:.2f} ± {r2_std:.2f}")
plt.text(0.95, 0.05, textstr,
         transform=plt.gca().transAxes,
         fontsize=12, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="lightgray", edgecolor="black"))

# === 5. Save figure ===================================================
plt.savefig(os.path.join(output_dir, "Scatter_PerSubject_STD_NoTrend.png"),
            dpi=300)
plt.close()









##########################################
# SCATTER WITH ±1 STD ERROR BARS PER SUBJECT + METRIC BOX
# Shows: mean prediction per subject + variability + metrics
##########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# === Create mean and std predictions per subject ===
df_stats = (df_preds
            .groupby("Subject_ID")
            .agg(Real_Age   = ('Real_Age',      'first'),
                 Pred_Mean  = ('Predicted_Age', 'mean'),
                 Pred_Std   = ('Predicted_Age', 'std'))
            .reset_index())

# === Create scatter plot with error bars ===
plt.figure(figsize=(8, 6))

# Plot: mean prediction ± 1 std per subject
plt.errorbar(df_stats["Real_Age"],
             df_stats["Pred_Mean"],
             yerr=df_stats["Pred_Std"],
             fmt='o',
             ecolor='gray',
             elinewidth=1,
             capsize=3,
             alpha=0.8,
             markeredgecolor='k',
             label="Mean ± 1 STD")

# Ideal diagonal y = x
min_age = min(df_stats["Real_Age"].min(), df_stats["Pred_Mean"].min())
max_age = max(df_stats["Real_Age"].max(), df_stats["Pred_Mean"].max())
margin  = (max_age - min_age) * 0.05
plt.plot([min_age, max_age], [min_age, max_age],
         ls='--', color='red', label='Ideal (y = x)')

# Linear regression fit
reg = LinearRegression().fit(df_stats["Real_Age"].values.reshape(-1, 1),
                             df_stats["Pred_Mean"].values)
trend_x = np.array([min_age, max_age]).reshape(-1, 1)
trend_y = reg.predict(trend_x)
plt.plot(trend_x, trend_y,
         color='blue', alpha=0.6, linewidth=2,
         label=f"Trend: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}")

# Axis limits
plt.xlim(min_age - margin, max_age + margin)
plt.ylim(min_age - margin, max_age + margin)

# Axis labels and title
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean ± std)")
plt.title("Predicted vs Real Ages — one dot per subject")
plt.legend(loc="upper left")
plt.grid(True)

# === Compute metrics across repetitions (on per-subject means) ===
repeats = 10  # adjust if you used a different number
maes, rmses, r2s = [], [], []

for rep in range(repeats):
    # Select rows for this repetition
    df_rep = df_preds.iloc[rep::repeats]  # assumes 10 blocks ordered by rep
    df_rep_mean = (df_rep.groupby("Subject_ID")
                           .agg({"Real_Age": "first",
                                 "Predicted_Age": "mean"}))
    # Compute metrics
    mae  = mean_absolute_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"])
    rmse = mean_squared_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"], squared=False)
    r2   = r2_score(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"])

    maes.append(mae)
    rmses.append(rmse)
    r2s.append(r2)

# === Metric box (mean ± std across repetitions) ===
mae_mean, mae_std   = np.mean(maes),  np.std(maes)
rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
r2_mean,   r2_std   = np.mean(r2s),   np.std(r2s)

# Add textbox with metrics
textstr = (f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}\n"
           f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
           f"R²:   {r2_mean:.2f} ± {r2_std:.2f}")
plt.text(0.95, 0.05, textstr,
         transform=plt.gca().transAxes,
         fontsize=12, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="lightgray", edgecolor="black"))

# Save the figure
plt.savefig(os.path.join(output_dir, "Scatter_PerSubject_withSTD_andMetrics.png"),
            dpi=300)
plt.close()




##########################################
# SCATTER PER SUBJECT + ±1 STD + Nicer Trend Line
##########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# === 1. Aggregate mean & std per subject ==============================
df_stats = (df_preds
            .groupby("Subject_ID")
            .agg(Real_Age   = ('Real_Age',      'first'),
                 Pred_Mean  = ('Predicted_Age', 'mean'),
                 Pred_Std   = ('Predicted_Age', 'std'))
            .reset_index())

# === 2. Figure setup =================================================
plt.figure(figsize=(8, 6))

# Scatter with vertical error bars (±1 STD)
plt.errorbar(df_stats["Real_Age"],
             df_stats["Pred_Mean"],
             yerr=df_stats["Pred_Std"],
             fmt='o',
             ecolor='gray',
             elinewidth=1,
             capsize=3,
             alpha=0.85,
             markeredgecolor='k',
             label="Mean ± 1 STD")

# Ideal y = x diagonal
min_age = min(df_stats["Real_Age"].min(), df_stats["Pred_Mean"].min())
max_age = max(df_stats["Real_Age"].max(), df_stats["Pred_Mean"].max())
margin  = (max_age - min_age) * 0.05
plt.plot([min_age, max_age], [min_age, max_age],
         ls='--', color='red', linewidth=1.0, label='Ideal (y = x)')

# === 3. Trend line (full-range, visually neat) =======================
# Fit linear regression on the mean points
reg = LinearRegression().fit(df_stats["Real_Age"].values.reshape(-1, 1),
                             df_stats["Pred_Mean"].values)

# Predict exactly at the X-axis limits to cover the whole plot
trend_x = np.array([min_age - margin, max_age + margin]).reshape(-1, 1)
trend_y = reg.predict(trend_x)

# Draw the line with slightly thicker width for clarity
plt.plot(trend_x, trend_y,
         color='#1f77b4', linewidth=2.5,  # same default blue but thicker
         label=f"Trend: y = {reg.coef_[0]:.2f} x + {reg.intercept_:.2f}")

# === 4. Styling =======================================================
plt.xlim(min_age - margin, max_age + margin)
plt.ylim(min_age - margin, max_age + margin)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean ± std)")
plt.title("Predicted vs Real Ages — per subject")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)

# === 5. Global metrics box ===========================================
repeats = 10  # change if different
maes, rmses, r2s = [], [], []

for rep in range(repeats):
    df_rep = df_preds.iloc[rep::repeats]          # rows for repetition `rep`
    df_rep_mean = (df_rep.groupby("Subject_ID")
                           .agg({"Real_Age": "first",
                                 "Predicted_Age": "mean"}))

    maes.append(mean_absolute_error(df_rep_mean["Real_Age"],
                                    df_rep_mean["Predicted_Age"]))
    rmses.append(mean_squared_error(df_rep_mean["Real_Age"],
                                    df_rep_mean["Predicted_Age"],
                                    squared=False))
    r2s.append(r2_score(df_rep_mean["Real_Age"],
                        df_rep_mean["Predicted_Age"]))

mae_mean, mae_std   = np.mean(maes),  np.std(maes)
rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
r2_mean,   r2_std   = np.mean(r2s),   np.std(r2s)

textstr = (f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}\n"
           f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
           f"R²:   {r2_mean:.2f} ± {r2_std:.2f}")
plt.text(0.95, 0.05, textstr,
         transform=plt.gca().transAxes,
         fontsize=12, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="lightgray", edgecolor="black"))

# === 6. Save figure ===================================================
plt.savefig(os.path.join(output_dir, "Scatter_PerSubject_STD_NiceTrend.png"),
            dpi=300)
plt.close()





#NEW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os

# === 1. Compute subject-level stats: mean and std per subject ===
df_stats = (
    df_preds.groupby("Subject_ID")
    .agg(Real_Age   = ('Real_Age', 'first'),
         Pred_Mean  = ('Predicted_Age', 'mean'),
         Pred_Std   = ('Predicted_Age', 'std'))
    .reset_index()
)

# === 2. Compute metrics over all 10 repetitions (repetition-aware) ===
repeats = 10  # update if using more
maes, rmses, r2s = [], [], []

for rep in range(repeats):
    df_rep = df_preds.iloc[rep::repeats]
    df_rep_mean = df_rep.groupby("Subject_ID").agg({
        "Real_Age": "first",
        "Predicted_Age": "mean"
    })
    maes.append(mean_absolute_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"]))
    rmses.append(mean_squared_error(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"], squared=False))
    r2s.append(r2_score(df_rep_mean["Real_Age"], df_rep_mean["Predicted_Age"]))

mae_mean, mae_std   = np.mean(maes),  np.std(maes)
rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
r2_mean,   r2_std   = np.mean(r2s),   np.std(r2s)

# === 3. Scatter plot with ±1 std and trend line ===
plt.figure(figsize=(8, 6))

# Points with error bars
plt.errorbar(df_stats["Real_Age"], df_stats["Pred_Mean"],
             yerr=df_stats["Pred_Std"],
             fmt='o', ecolor='gray', elinewidth=1, capsize=3,
             alpha=0.85, markeredgecolor='k', label="Mean ± 1 STD")

# Ideal diagonal
min_age = df_stats[["Real_Age", "Pred_Mean"]].min().min() - 5
max_age = df_stats[["Real_Age", "Pred_Mean"]].max().max() + 5
plt.plot([min_age, max_age], [min_age, max_age],
         ls='--', color='red', linewidth=1.0, label='Ideal (y = x)')

# Trend line
reg = LinearRegression().fit(
    df_stats["Real_Age"].values.reshape(-1, 1),
    df_stats["Pred_Mean"].values
)
x_vals = np.array([min_age, max_age]).reshape(-1, 1)
y_vals = reg.predict(x_vals)
plt.plot(x_vals, y_vals,
         color="blue", alpha=0.6, linewidth=2,
         label=f"Trend: y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}")

# Axes
plt.xlim(min_age, max_age)
plt.ylim(min_age, max_age)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age (mean ± std)")
plt.title("Predicted vs Real Age — per subject (±1 STD + trend)")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")

# Metrics box: across 10 repetitions
textstr = (f"MAE:  {mae_mean:.2f} ± {mae_std:.2f}\n"
           f"RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}\n"
           f"R²:   {r2_mean:.2f} ± {r2_std:.2f}")
plt.text(0.95, 0.05, textstr,
         transform=plt.gca().transAxes,
         fontsize=12, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black"))

# Save
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "New_Scatter_PerSubject_STD_Trend_Final.png"), dpi=300)
plt.close()








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
final_model = BrainAgeGATv2().to(device)

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



