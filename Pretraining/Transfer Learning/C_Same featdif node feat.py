#Full Fine Tuning ADNI to ADDECODE Same Features


#ADNI-> Node features: FA, VOL​
#Features : Sex, APOE​
#AADCODE-> Node features: FA VOL MD​
#Features: Sex, APOE​

# ==== BLOCK 1: SETUP AND IMPORTS ====

# Standard libraries
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import re

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# NetworkX for graph metrics
import networkx as nx

# ==== Set seed for reproducibility ====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==== Check if CUDA is available ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")



# ==== BLOCK 2: GATv2 MODEL FOR AD-DECODE ====

class BrainAgeGATv2(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2, self).__init__()

        # Node embedding: input_dim = 3 (FA, MD, Volume)
        self.node_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # GATv2 layers with skip connections
        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

        # Final MLP: input = pooled graph features + global features
        self.fc = nn.Sequential(
            nn.Linear(128 + global_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.node_embed(x)

        x = self.gnn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)  # Pooled graph representation

        global_feats = data.global_features.to(x.device)
        x = torch.cat([x, global_feats], dim=1)  # Concatenate with global features

        x = self.fc(x)
        return x



# ==== BLOCK 3: LOAD PRETRAINED WEIGHTS (TRANSFER LEARNING) ====

# Step 1: Initialize the AD-DECODE model
model = BrainAgeGATv2(global_feat_dim=6).to(device)

# Load the pretrained weights
pretrained_weights = torch.load("brainage_adni_pretrained.pt", map_location=device)

# Delete node_embed layer weights because now we have 3 node features instead of 2
keys_to_remove = [k for k in pretrained_weights.keys() if "node_embed" in k]
for key in keys_to_remove:
    del pretrained_weights[key]

# Now load weights without error
model.load_state_dict(pretrained_weights, strict=False)




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

print("Example of extracted connectome numbers:")
for key in list(cleaned_connectomes.keys())[:3]:
    print(key)
print()

############################## METADATA ##############################


print("Addecode metadata\n")

# === Load metadata CSV ===
metadata_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data_defaced.csv"
df_metadata = pd.read_csv(metadata_path)

# === Generate standardized subject IDs → 'DWI_fixed' (e.g., 123 → '00123')
df_metadata["DWI_fixed"] = (
    df_metadata["DWI"]
    .fillna(0)                           # Handle NaNs first
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# === Drop fully empty rows and those with missing DWI ===
df_metadata_cleaned = df_metadata.dropna(how='all')                       # Remove fully empty rows
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["DWI"])         # Remove rows without DWI

# === Display result ===
print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()
print("Example of 'DWI_fixed' column:")
print(df_metadata_cleaned[["DWI", "DWI_fixed"]].head())
print()



#################### MATCH CONNECTOMES & METADATA ####################

print("### MATCHING CONNECTOMES WITH METADATA")

# === Filter metadata to only subjects with connectomes available ===
matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["DWI_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

# === Build dictionary of matched connectomes ===
matched_connectomes = {
    row["DWI_fixed"]: cleaned_connectomes[row["DWI_fixed"]]
    for _, row in matched_metadata.iterrows()
}


# === Store matched metadata as a DataFrame for further processing ===
df_matched_connectomes = matched_metadata.copy()


#################### SHOW EXAMPLE CONNECTOME WITH AGE ####################

# === Display one matched connectome and its metadata ===
example_id = df_matched_connectomes["DWI_fixed"].iloc[0]
example_age = df_matched_connectomes["age"].iloc[0]
example_matrix = matched_connectomes[example_id]

print(f"Example subject ID: {example_id}")
print(f"Age: {example_age}")
print("Connectome matrix (first 5 rows):")
print(example_matrix.head())
print()

# === Plot heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(example_matrix, cmap="viridis")
plt.title(f"Connectome Heatmap - Subject {example_id} (Age {example_age})")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

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

print(f"Subjects before filtering: {len(df_matched_connectomes)}")
print(f"Subjects after removing AD/MCI: {len(df_matched_addecode_healthy)}")

# === Show updated 'Risk' distribution ===
if "Risk" in df_matched_addecode_healthy.columns:
    risk_filled = df_matched_addecode_healthy["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()




# === Filter connectomes to include only those from non-AD/MCI subjects ===
matched_connectomes_healthy_addecode = {
    row["DWI_fixed"]: matched_connectomes[row["DWI_fixed"]]
    for _, row in df_matched_addecode_healthy.iterrows()
}

# === Confirmation of subject count
print(f"Connectomes selected (excluding AD/MCI): {len(matched_connectomes_healthy_addecode)}")



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





#################  PREPROCESS DEMOGRAPHIC FEATURES (NO SCALING / RAW INPUT)  ################

from sklearn.preprocessing import LabelEncoder
import torch

# === Reset index ===
addecode_healthy_metadata = df_matched_addecode_healthy.reset_index(drop=True)


# === Define selected feature groups (reduced) ===

categorical_label_cols = ["sex"]             # label encode
categorical_ordered_cols = ["genotype"]      # label encode

# === Drop rows with missing values in selected columns ===
all_required_cols = categorical_label_cols + categorical_ordered_cols
addecode_healthy_metadata = addecode_healthy_metadata.dropna(subset=all_required_cols).reset_index(drop=True)

# === Label encode binary categorical (sex) ===
for col in categorical_label_cols:
    le = LabelEncoder()
    addecode_healthy_metadata[col] = le.fit_transform(addecode_healthy_metadata[col].astype(str))

# === Label encode ordered categorical (genotype) ===
for col in categorical_ordered_cols:
    le = LabelEncoder()
    addecode_healthy_metadata[col] = le.fit_transform(addecode_healthy_metadata[col].astype(str))

# === Build metadata DataFrame ===
meta_df = addecode_healthy_metadata[categorical_label_cols + categorical_ordered_cols]

# === Convert to float and build subject dictionary ===
meta_df = meta_df.astype(float)

subject_to_meta_addecode = {
    row["DWI_fixed"]: torch.tensor(meta_df.values[i], dtype=torch.float)
    for i, row in addecode_healthy_metadata.iterrows()
}



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

# === Normalización nodo-wise entre sujetos ===
def normalize_multimodal_nodewise(feature_dict):
    all_features = torch.stack(list(feature_dict.values()))  # [N_subjects, 84, 3]
    means = all_features.mean(dim=0)  # [84, 3]
    stds = all_features.std(dim=0) + 1e-8
    return {subj: (features - means) / stds for subj, features in feature_dict.items()}

# Aplica normalización
normalized_node_features_dict = normalize_multimodal_nodewise(multimodal_features_dict)




import numpy as np
import pandas as pd

def threshold_connectome(matrix, percentile=100):
    """
    Apply percentile-based thresholding to a connectome matrix.

    Parameters:
    - matrix (pd.DataFrame): The original connectome matrix (84x84).
    - percentile (float): The percentile threshold to keep. 
                          100 means keep all, 75 means keep top 75%, etc.

    Returns:
    - thresholded_matrix (pd.DataFrame): A new matrix with only strong connections kept.
    """
    
    # === 1. Flatten the matrix and exclude diagonal (self-connections) ===
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)  # Mask to exclude diagonal
    values = matrix_np[mask]  # Get all off-diagonal values

    # === 2. Compute the threshold value based on percentile ===
    threshold_value = np.percentile(values, 100 - percentile)

    # === 3. Apply thresholding: keep only values >= threshold, set others to 0 ===
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)

    # === 4. Return as DataFrame with same structure ===
    thresholded_matrix = pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)
    return thresholded_matrix





#####################  APPLY THRESHOLD + LOG TRANSFORM #######################

log_thresholded_connectomes = {}

for subject, matrix in matched_connectomes_healthy_addecode.items():

    # === 1. Apply 95% threshold ===
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
    
    # === 2. Apply log(x + 1) ===
    log_matrix = np.log1p(thresholded_matrix)
    
    # === 3. Store matrix with same shape and index ===
    log_thresholded_connectomes[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)



# Visual check of first 3 transformed matrices
for subject, matrix_log in list(log_thresholded_connectomes.items())[:1]:
    print(f"Log-transformed matrix for Subject {subject}:")
    print(matrix_log)
    print()  # Imprimir una línea vacía para separar



##################### MATRIX TO GRAPH #######################

import torch
import numpy as np
from torch_geometric.data import Data


# === Function to convert a connectome matrix into a graph with multimodal node features ===
def matrix_to_graph(matrix, device, subject_id, node_features_dict):
    indices = np.triu_indices(84, k=1)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long, device=device)
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float, device=device)

    # Usa features normalizadas multimodales
    node_feats = node_features_dict[subject_id]  # shape [84, 2]
    node_features =  0.5 * node_feats.to(device)  # Optional scaling
    
    return edge_index, edge_attr, node_features

    
    
    
    
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

################# GLOBAL EFFICIENCY ############################

def compute_global_efficiency(matrix):
    """
    Computes the global efficiency of a graph from a connectome matrix.
    
    Parameters:
    - matrix (pd.DataFrame): 84x84 connectivity matrix
    
    Returns:
    - float: global efficiency
    """
    G = nx.from_numpy_array(matrix.to_numpy())

    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]

    return nx.global_efficiency(G)


############################## LOCAL EFFICIENCY #######################

def compute_local_efficiency(matrix):
    """
    Computes the local efficiency of the graph.
    
    Parameters:
    - matrix (pd.DataFrame): 84x84 connectivity matrix
    
    Returns:
    - float: local efficiency
    """
    G = nx.from_numpy_array(matrix.to_numpy())

    for u, v, d in G.edges(data=True):
        d["weight"] = matrix.iloc[u, v]

    return nx.local_efficiency(G)




#####################  DEVICE CONFIGURATION  #######################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


print(f"Total connectomes after filtering: {len(matched_connectomes_healthy_addecode)}")
print(f"Total node features available: {len(normalized_node_features_dict)}")
print(f"Total metadata available: {len(subject_to_meta_addecode)}")

#################  CONVERT MATRIX TO GRAPH  ################

graph_data_list_addecode = []

for subject, matrix_log in log_thresholded_connectomes.items():
    if subject not in subject_to_meta_addecode:
        continue  # Skip if the subject does not have the demographic feature
    
    if subject not in normalized_node_features_dict:
        continue  # Skip if no node features
    
    
    # === Convert matrix to graph ===
    edge_index, edge_attr, node_features = matrix_to_graph(matrix_log, device, subject, normalized_node_features_dict)



    # === Get age as target ===
    age_row = df_matched_addecode_healthy.loc[df_matched_addecode_healthy["DWI_fixed"] == subject, "age"]


    
    if not age_row.empty:
        age = torch.tensor([age_row.values[0]], dtype=torch.float)



        # === Compute graph metrics ===
        
        clustering_coeff = compute_clustering_coefficient(matrix_log)
        path_length = compute_path_length(matrix_log)
        global_eff = compute_global_efficiency(matrix_log)
        local_eff = compute_local_efficiency(matrix_log)
        
        graph_metrics_tensor = torch.tensor(
            [clustering_coeff, path_length, global_eff, local_eff], dtype=torch.float
        )


        
                
        # === Append graph metrics to demographic metadata ===
        base_meta = subject_to_meta_addecode[subject]
        global_feat = torch.cat([base_meta, graph_metrics_tensor], dim=0)

        

        # === Create Data object with global feature ===
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0)  # Shape becomes (1, num_features)
        )

        graph_data_list_addecode.append(data)



for i, data in enumerate(graph_data_list_addecode[:1]):
    subject_id = addecode_healthy_metadata.iloc[i]["DWI_fixed"]
    age = addecode_healthy_metadata.iloc[i]["age"]
    print(f"{i+1}. Subject: {subject_id}, Age: {age}, Target y: {data.y.item()}")


# Display the first graph's data structure for verification
print(f"Example graph structure: {graph_data_list_addecode[0]}")

print(f" Total graphs ready for training: {len(graph_data_list_addecode)}")


# ==== BLOCK 5: Training and Evaluation Functions ====

from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score

# Define training function
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)  # Move batch to GPU
        optimizer.zero_grad()  # Reset gradients
        output = model(data).view(-1)  # Predict
        loss = criterion(output, data.y)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Define evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # No gradient computation during evaluation
        for data in test_loader:
            data = data.to(device)
            output = model(data).view(-1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)



# ==== BLOCK 6: Cross-validation Setup + Progressive Unfreezing ====

from torch_geometric.loader import DataLoader

# Define training hyperparameters
epochs = 300
patience = 30
k = 7  # Number of folds
repeats_per_fold = 10
batch_size = 6

# Prepare age bins for stratified split
ages = df_matched_addecode_healthy["age"].to_numpy()
age_bins = pd.qcut(ages, q=5, labels=False)

# Set up stratified K-Fold
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize storage for all metrics
all_train_losses = []
all_test_losses = []
all_early_stopping_epochs = []
all_y_true = []
all_y_pred = []
fold_mae_list = []
fold_r2_list = []

# Start Cross-Validation loop
for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f"\n--- Fold {fold+1}/{k} ---")

    train_data = [graph_data_list_addecode[i] for i in train_idx]
    test_data = [graph_data_list_addecode[i] for i in test_idx]

    fold_train_losses = []
    fold_test_losses = []

    for repeat in range(repeats_per_fold):
        print(f"  > Repeat {repeat+1}/{repeats_per_fold}")

        seed_everything(42 + repeat)  # Ensure reproducibility for each repeat

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # === Initialize model ===
        model = BrainAgeGATv2(global_feat_dim=6).to(device)

        # === Load pretrained ADNI weights  ===
        
        pretrained_weights = torch.load("brainage_adni_pretrained.pt", map_location=device)
        
        # Delete node_embed layer weights because now we have 3 node features instead of 2
        keys_to_remove = [k for k in pretrained_weights.keys() if "node_embed" in k]
        for key in keys_to_remove:
            del pretrained_weights[key]
        
        # Now load weights
        model.load_state_dict(pretrained_weights, strict=False)



        
        optimizer = AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = torch.nn.SmoothL1Loss(beta=1)

        best_loss = float('inf')
        patience_counter = 0
        early_stop_epoch = None

        train_losses = []
        test_losses = []

        for epoch in range(epochs):

            
            # === Training step ===
            train_loss = train(model, train_loader, optimizer, criterion)
        
            # === Evaluation step ===
            test_loss = evaluate(model, test_loader, criterion)
        
            # === Store losses for plotting later ===
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
            # === Early stopping check ===
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"full_Fine_Tuning2_fold_{fold+1}_rep_{repeat+1}.pt")  # save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_epoch = epoch + 1
                    print(f"    Early stopping triggered at epoch {early_stop_epoch}.")
                    break
        
            # === Step the learning rate scheduler ===
            scheduler.step()


        if early_stop_epoch is None:
            early_stop_epoch = epochs
        all_early_stopping_epochs.append((fold + 1, repeat + 1, early_stop_epoch))

        fold_train_losses.append(train_losses)
        fold_test_losses.append(test_losses)

    all_train_losses.append(fold_train_losses)
    all_test_losses.append(fold_test_losses)



#  LEARNING CURVE PLOT ====


# === Import necessary libraries for plotting ===
import matplotlib.pyplot as plt

# === Create a new figure ===
plt.figure(figsize=(10, 6))

# === Plot learning curves for each fold and repetition ===
for fold in range(k):
    for rep in range(repeats_per_fold):
        plt.plot(all_train_losses[fold][rep], label=f'Train Loss - Fold {fold+1} Rep {rep+1}', linestyle='dashed', alpha=0.5)
        plt.plot(all_test_losses[fold][rep], label=f'Test Loss - Fold {fold+1} Rep {rep+1}', alpha=0.5)

# === Add labels and title ===
plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning Curves (All Repeats) Full Fine Tuning Same Features for ADNI and ADDECODE")

# === Add legend and grid ===
plt.legend(loc="upper right", fontsize=8)
plt.grid(True)

# === Show the plot ===
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
plt.title("Learning Curve (Mean ± Std Across All Folds/Repeats) Full Fine Tuning Same Features for ADNI and ADDECODE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# FINAL EVALUATION AND PLOT 

# === Import metrics libraries ===
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# === Initialize lists to store results ===
fold_mae_list = []
fold_r2_list = []
all_y_true = []
all_y_pred = []

# === Evaluate each fold and repetition ===
for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_addecode, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    # === Create test set ===
    test_data = [graph_data_list_addecode[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        # === Load the best model saved for this fold and repeat ===
        model = BrainAgeGATv2(global_feat_dim=6).to(device)
        model.load_state_dict(torch.load(f"full_Fine_Tuning2_fold_{fold+1}_rep_{rep+1}.pt"))
        model.eval()

        # === Predict ===
        y_true_repeat = []
        y_pred_repeat = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data).view(-1)
                y_pred_repeat.extend(output.cpu().tolist())
                y_true_repeat.extend(data.y.cpu().tolist())

        # === Calculate metrics for this repeat ===
        mae = mean_absolute_error(y_true_repeat, y_pred_repeat)
        r2 = r2_score(y_true_repeat, y_pred_repeat)

        repeat_maes.append(mae)
        repeat_r2s.append(r2)

        # === Store all predictions ===
        all_y_true.extend(y_true_repeat)
        all_y_pred.extend(y_pred_repeat)

    fold_mae_list.append(repeat_maes)
    fold_r2_list.append(repeat_r2s)

    print(f">> Fold {fold+1} | MAE: {np.mean(repeat_maes):.2f} ± {np.std(repeat_maes):.2f} | R²: {np.mean(repeat_r2s):.2f} ± {np.std(repeat_r2s):.2f}")

# === Aggregate final metrics ===
all_maes = np.array(fold_mae_list).flatten()
all_r2s = np.array(fold_r2_list).flatten()

print("\n================== FINAL METRICS ==================")
print(f"Global MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f}")
print(f"Global R²:  {np.mean(all_r2s):.2f} ± {np.std(all_r2s):.2f}")
print("===================================================")




# ==== PLOT TRUE VS PREDICTED AGES ====

plt.figure(figsize=(8, 6))
plt.scatter(all_y_true, all_y_pred, alpha=0.7, edgecolors='k', label="Predictions")

min_val = min(min(all_y_true), min(all_y_pred))
max_val = max(max(all_y_true), max(all_y_pred))
margin = (max_val - min_val) * 0.05

plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal (y = x)")
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

# Metrics box
textstr = f"MAE: {np.mean(all_maes):.2f} ± {np.std(all_maes):.2f}\nR²: {np.mean(all_r2s):.2f} ± {np.std(all_r2s):.2f}"
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs Real Age: Full Fine Tuning Same Features for ADNI and ADDECODE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


