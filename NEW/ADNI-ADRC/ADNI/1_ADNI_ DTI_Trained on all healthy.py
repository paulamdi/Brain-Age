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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import networkx as nx
from torch_geometric.data import Data

# === Get valid subjects (those with both connectome and metadata matched) ===
valid_subjects = set(df_matched_adni_healthy["connectome_key"])

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
log_thresholded_connectomes_adni = {}
for subject, matrix in matched_connectomes_healthy_adni.items():
    thresholded_matrix = threshold_connectome(matrix, percentile=95)
    log_matrix = np.log1p(thresholded_matrix)
    log_thresholded_connectomes_adni[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)

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
adni_healthy_metadata = df_matched_adni_healthy.reset_index(drop=True)
adni_healthy_metadata["Clustering_Coeff"] = np.nan
adni_healthy_metadata["Path_Length"] = np.nan
adni_healthy_metadata["Global_Efficiency"] = np.nan
adni_healthy_metadata["Local_Efficiency"] = np.nan

for subject, matrix_log in log_thresholded_connectomes_adni.items():
    try:
        clustering = compute_clustering_coefficient(matrix_log)
        path = compute_path_length(matrix_log)
        global_eff = compute_global_efficiency(matrix_log)
        local_eff = compute_local_efficiency(matrix_log)

        adni_healthy_metadata.loc[adni_healthy_metadata["connectome_key"] == subject, [
            "Clustering_Coeff", "Path_Length", "Global_Efficiency", "Local_Efficiency"
        ]] = [clustering, path, global_eff, local_eff]
    except Exception as e:
        print(f"Failed to compute metrics for subject {subject}: {e}")





# === Label encode and normalize ===
adni_healthy_metadata["sex_encoded"] = LabelEncoder().fit_transform(adni_healthy_metadata["Sex"].astype(str))
adni_healthy_metadata["genotype"] = LabelEncoder().fit_transform(
    adni_healthy_metadata["APOE A1"].astype(str) + "_" + adni_healthy_metadata["APOE A2"].astype(str)
)

numerical_cols = ["Clustering_Coeff", "Path_Length", "Global_Efficiency", "Local_Efficiency"]
adni_healthy_metadata[numerical_cols] = adni_healthy_metadata[numerical_cols].apply(zscore)








# === Build global feature tensors ===
subject_to_demographic_tensor = {
    row["connectome_key"]: torch.tensor([
        row["sex_encoded"],
        row["genotype"]
    ], dtype=torch.float)
    for _, row in adni_healthy_metadata.iterrows()
}

subject_to_graphmetric_tensor = {
    row["connectome_key"]: torch.tensor([
        row["Clustering_Coeff"],
        row["Path_Length"],
        row["Global_Efficiency"],
        row["Local_Efficiency"]
    ], dtype=torch.float)
    for _, row in adni_healthy_metadata.iterrows()
}








#####################  DEVICE CONFIGURATION  #######################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



#################  CONVERT MATRIX TO GRAPH  ################

from torch_geometric.data import Data

graph_data_list_adni = []

for subject, matrix_log in log_thresholded_connectomes_adni.items():
    try:
        # Skip if missing required components
        if subject not in subject_to_demographic_tensor:
            continue
        if subject not in subject_to_graphmetric_tensor:
            continue
        if subject not in normalized_node_features_dict:
            continue

        # Convert connectome matrix to graph
        edge_index, edge_attr, node_features = matrix_to_graph(
            matrix_log, device, subject, normalized_node_features_dict
        )

        # Get subject age
        age_row = df_matched_adni_healthy[
            df_matched_adni_healthy["connectome_key"] == subject
        ]["Age"]
        if age_row.empty:
            continue
        age = torch.tensor([age_row.values[0]], dtype=torch.float)

        # Concatenate demographic + graph metrics → global features
        demo_tensor = subject_to_demographic_tensor[subject]  # [2]
        graph_tensor = subject_to_graphmetric_tensor[subject]  # [2]
        global_feat = torch.cat([demo_tensor, graph_tensor], dim=0)  # [4]

        # Create graph object
        data = Data(
            x=node_features,  # [84, 2]
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=age,
            global_features=global_feat.unsqueeze(0)  # [1, 4]
        )
        data.subject_id = subject
        graph_data_list_adni.append(data)

    except Exception as e:
        print(f"Failed to process subject {subject}: {e}")


# Print basic info for one graph
sample_graph = graph_data_list_adni[0]
print("=== ADNI Sample Graph ===")
print(f"Node feature shape (x): {sample_graph.x.shape}")         
print(f"Edge index shape: {sample_graph.edge_index.shape}")     
print(f"Edge attr shape: {sample_graph.edge_attr.shape}")       
print(f"Global features shape: {sample_graph.global_features.shape}")  
print(f"Target age (y): {sample_graph.y.item()}")                

# Optional previews
print("\nFirst 5 edge weights:")
print(sample_graph.edge_attr[:5])

print("\nGlobal features vector:")
print(sample_graph.global_features)




######################  DEFINE MODEL  #########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm

class BrainAgeGATv2_ADNI(nn.Module):
    def __init__(self, global_feat_dim):
        super(BrainAgeGATv2_ADNI, self).__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # GATv2 layers using edge_attr (edge_dim=1)
        self.gnn1 = GATv2Conv(64, 16, heads=8, concat=True, edge_dim=1)
        self.bn1 = BatchNorm(128)

        self.gnn2 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn2 = BatchNorm(128)

        self.gnn3 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn3 = BatchNorm(128)

        self.gnn4 = GATv2Conv(128, 16, heads=8, concat=True, edge_dim=1)
        self.bn4 = BatchNorm(128)

        self.dropout = nn.Dropout(0.25)

        

        # === GLOBAL FEATURE BRANCHES ===
        # Demographics: sex, genotype (2 features)
        self.meta_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Graph metrics: clustering, path length, global/local efficiency (4 features)
        self.graph_head = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion will now be: 16 + 32 = 48
        self.fc = nn.Sequential(
            nn.Linear(128 + 48, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_embed(x)

        x = self.gnn1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x_res1 = x
        x = self.gnn2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x + x_res1)

        x_res2 = x
        x = self.gnn3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x + x_res2)

        x_res3 = x
        x = self.gnn4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.relu(x + x_res3)

        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)

        # Global feats: [batch, 1, 6]
        global_feats = data.global_features.to(x.device).squeeze(1)
        
        meta_embed = self.meta_head(global_feats[:, 0:2])   # Sex, genotype
        graph_embed = self.graph_head(global_feats[:, 2:])  # 4 graph metrics
        
        global_embed = torch.cat([meta_embed, graph_embed], dim=1)  # [batch, 48]
        x = torch.cat([x, global_embed], dim=1)
        x = self.fc(x)

        return x


    
    
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



# === Extract ages from metadata and create stratification bins ===
ages = df_matched_adni_healthy["Age"].to_numpy()

# === Create age bins for stratification ===
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

        model = BrainAgeGATv2_ADNI(global_feat_dim=6).to(device)  

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


for fold, (train_idx, test_idx) in enumerate(skf.split(graph_data_list_adni, age_bins)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_data = [graph_data_list_adni[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    repeat_maes = []
    repeat_r2s = []

    for rep in range(repeats_per_fold):
        print(f"  > Repeat {rep+1}/{repeats_per_fold}")

        model = BrainAgeGATv2_ADNI(global_feat_dim=6).to(device)  

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

from torch_geometric.loader import DataLoader

# === Full training with all ADNI data ===
train_loader = DataLoader(graph_data_list_adni, batch_size=6, shuffle=True)

# Initialize model and optimizer
model = BrainAgeGATv2_ADNI(global_feat_dim=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = torch.nn.SmoothL1Loss(beta=1)

# === Training loop ===


epochs = 150 

# seeing folds from cv
#Mediana de early stopping: ~95

#Media aproximada: ~100

#Percentil 75: ~120–130

#Máximo: 215 


train_losses = []

print("\n=== Full Training on ADNI (No Validation) ===")
for epoch in range(epochs):
    loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(loss)
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f}")

# === Save final model ===
torch.save(model.state_dict(), "brainage_adni_pretrained.pt")
print("\n Pretrained model saved as 'brainage_adni_pretrained.pt'")

# === Plot learning curve ===
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve - ADNI Pretraining")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()






model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for data in train_loader:
        data = data.to(device)
        pred = model(data).view(-1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(data.y.cpu().tolist())

mae = mean_absolute_error(all_labels, all_preds)
print(f"MAE en entrenamiento completo: {mae:.2f}")

