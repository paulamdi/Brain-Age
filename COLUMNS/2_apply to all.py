# COLUMNS MEAN



import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions

import torch
import random
import numpy as np

import os
import pandas as pd
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



#Load columns



# Base directory where subject folders (e.g., S00775_columns_1mm) are stored
BASE_DIR = "/home/bas/Desktop/MyData/Columns/input_columns"


# List of hemispheres
HEMIS = ["lh", "rh"]

# List of cortical regions (based on the parcellation used)
REGION_NAMES = ['bankssts',
'caudalanteriorcingulate',
'caudalmiddlefrontal',
'cuneus',
'entorhinal',
'fusiform',
'inferiorparietal',
'inferiortemporal',
'isthmuscingulate',
'lateraloccipital',
'lateralorbitofrontal',
'lingual',
'medialorbitofrontal',
'middletemporal',
'parahippocampal',
'paracentral',
'parsopercularis',
'parsorbitalis',
'parstriangularis',
'pericalcarine',
'postcentral',
'posteriorcingulate',
'precentral',
'precuneus',
'rostralanteriorcingulate',
'rostralmiddlefrontal',
'superiorfrontal',
'superiorparietal',
'superiortemporal',
'supramarginal',
'frontalpole',
'temporalpole',
'transversetemporal',
'insula'
]



# Paths to CSVs
SUBJECT_IDS_CSV = "/home/bas/Desktop/MyData/Columns/subject_ids.csv"
AGE_CSV = "/home/bas/Desktop/MyData/Columns/ages.csv"

# Load both files (no headers)
subject_df = pd.read_csv(SUBJECT_IDS_CSV, header=None)
age_df = pd.read_csv(AGE_CSV, header=None)

# Zero-pad subject IDs to 5 digits
subject_ids = subject_df[0].astype(str).apply(lambda x: x.zfill(5)).tolist()
ages = age_df[0].astype(float).tolist()

# Build dictionary: subject_id → age
age_dict = dict(zip(subject_ids, ages))



# Final dictionary: subject_id -> region -> matrix
column_data = {}

# Loop through subjects
for subj_id in subject_ids:
    subj_dict = {}
    subj_dir = os.path.join(BASE_DIR, f"S{subj_id}_columns_1mm")

    for hemi in HEMIS:
        for region in REGION_NAMES:
            region_key = f"{hemi}_{region}"
            filename = f"S{subj_id}_{region_key}_md.csv"
            filepath = os.path.join(subj_dir, filename)

            if not os.path.exists(filepath):
                print(f"[WARNING] Missing: {filepath}")
                continue

            mat = pd.read_csv(filepath, header=None).values.astype(np.float32)
            subj_dict[region_key] = mat

    if len(subj_dict) > 0:
        column_data[subj_id] = subj_dict
    else:
        print(f"[INFO] Skipping subject {subj_id} (no valid regions)")

# === Example: print region shapes per subject ===
for subj_id, regions in column_data.items():
    print(f"\nSubject {subj_id}:")
    for region_name, matrix in regions.items():
        print(f"  {region_name} → {matrix.shape}")




# CONSTRUCT GRAPHS 

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors



# Get subject's data (dict: region → matrix)
subject_data = column_data[subj_id]




#Normalize
from sklearn.preprocessing import StandardScaler

# Concatenate all regions for one subject
x_np = np.vstack(list(subject_data.values()))  # shape [n_total_columns, 21]

# Z-score normalization across all features (columns)
scaler = StandardScaler()
x_np = scaler.fit_transform(x_np)





# === STEP 1: Build edge_index from K-Nearest Neighbors ===

def build_knn_edge_index(x_np, k=10):
    """
    Builds the edge_index for a graph using KNN based on node features.

    Args:
        x_np (np.ndarray): shape [n_nodes, n_features]
        k (int): number of neighbors per node

    Returns:
        edge_index (torch.LongTensor): shape [2, num_edges]
    """
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(x_np)
    knn_graph = knn.kneighbors_graph(x_np).tocoo()

    edge_index = torch.tensor(
        np.vstack((knn_graph.row, knn_graph.col)),
        dtype=torch.long
    )
    return edge_index

# === STEP 2: Build PyG graph from a single subject ===

def build_graph_from_subject(subj_id, column_data, age_dict, k=10):
    """
    Converts one subject's cortical column data into a PyTorch Geometric graph.

    Args:
        subj_id (str): subject ID
        column_data (dict): {subject_id → {region_name → matrix}}
        age_dict (dict): {subject_id → age}
        k (int): number of KNN neighbors

    Returns:
        Data: PyTorch Geometric graph with x, edge_index, y
    """
    # Get all regions from this subject and stack all columns
    subj_regions = column_data[subj_id]
    x_np = np.vstack(list(subj_regions.values()))  # shape [n_columns_total, 21]

    # Convert features to tensor
    x = torch.tensor(x_np, dtype=torch.float32)

    # Build edge_index using KNN over node features
    edge_index = build_knn_edge_index(x_np, k=k)

    # Get subject's age and convert to tensor
    y = torch.tensor([age_dict[subj_id]], dtype=torch.float32)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.subject_id = subj_id  # optional: store ID for reference
    return data

# === STEP 3: Loop through all subjects and build their graphs ===

graph_data_list_all = []

for subj_id in column_data:
    if subj_id in age_dict:
        graph = build_graph_from_subject(subj_id, column_data, age_dict, k=10)
        graph_data_list_all.append(graph)
        print(f" Built graph for subject {subj_id} with {graph.num_nodes} nodes")
    else:
        print(f"[WARNING] No age found for subject {subj_id}")




#Visualize graph

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# === Pick the first subject's graph ===
graph = graph_data_list_all[0]

# === Convert PyG graph to NetworkX for visualization ===
G = to_networkx(graph, to_undirected=True)

# === Optional: layout (slower for large graphs) ===
pos = nx.spring_layout(G, seed=42)  # force-directed layout

# === Plot ===
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=10, node_color='steelblue', edge_color='gray', alpha=0.6, width=0.5)

plt.title(f"Subject {graph.subject_id} – {graph.num_nodes} nodes")
plt.axis('off')
plt.show()




#Apply pretraining


#####################  DEVICE CONFIGURATION  #######################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



#DEFINE MODEL GCNs
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class BrainAgeGNN(torch.nn.Module):
    def __init__(self, input_dim=21):  # ← set to 21 features per node
        super(BrainAgeGNN, self).__init__()

        self.conv1 = GCNConv(input_dim, 64)
        self.bn1 = BatchNorm(64)

        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)

        self.conv3 = GCNConv(128, 128)
        self.bn3 = BatchNorm(128)

        self.fc = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x_residual = x
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = x + x_residual

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)

        return x.view(-1)






from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import torch

# === Load the trained model on all healthy+familial subjects ===
model = BrainAgeGNN(input_dim=21).to(device)  # Input: 21 features per column
model.load_state_dict(torch.load("column_md_model_trained_on_all_healthy.pt"))  # Load pretrained weights
model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# === Create a DataLoader for all subjects (including AD, MCI, etc.) ===
all_graph_loader = DataLoader(graph_data_list_all, batch_size=6, shuffle=False)

# === Prepare storage for results ===
all_preds = []        # Predicted ages
all_ids = []          # Subject IDs
all_true_ages = []    # Ground-truth chronological ages

# === Perform inference ===
with torch.no_grad():  # Disable gradient computation for efficiency
    for data in all_graph_loader:
        data = data.to(device)  # Move data to GPU or CPU
        output = model(data).view(-1)  # Predict age for each graph in the batch
        all_preds.extend(output.cpu().tolist())        # Store predictions
        all_true_ages.extend(data.y.cpu().tolist())    # Store true ages
        all_ids.extend(data.subject_id)                # Store subject IDs

# === Organize results into a DataFrame ===
df_results = pd.DataFrame({
    "subject_id": all_ids,
    "age": all_true_ages,
    "predicted_age": all_preds
})

# === Compute Brain Age Gap (BAG = predicted - actual age) ===
df_results["BAG"] = df_results["predicted_age"] - df_results["age"]

# === Save results to CSV file ===
df_results.to_csv("column_model_predictions_all_subjects.csv", index=False)
print(" Predictions saved to 'column_model_predictions_all_subjects.csv'")





#cbag

from sklearn.linear_model import LinearRegression
import numpy as np

# === Compute corrected BAG (cBAG) ===

# Fit linear regression: BAG ~ age
X = df_results["age"].values.reshape(-1, 1)  # Predictor: age
y = df_results["BAG"].values                # Target: BAG
reg = LinearRegression().fit(X, y)

# Predict age effect (regression line)
age_effect = reg.predict(X)

# Subtract regression line to get cBAG
df_results["cBAG"] = df_results["BAG"] - age_effect

# === Save updated results ===
df_results.to_csv("column_model_predictions_all_subjects_with_cBAG.csv", index=False)
print(" Results with cBAG saved to 'column_model_predictions_all_subjects_with_cBAG.csv'")



#plot bag and cbag

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Set plot style
sns.set(style="whitegrid")

# ========================== Plot 1: BAG vs Age ==========================
plt.figure(figsize=(8, 6))

# Scatter plot
sns.scatterplot(x="age", y="BAG", data=df_results, color="orange", alpha=0.6, edgecolor="k")

# Trendline
X_age = df_results["age"].values.reshape(-1, 1)
y_bag = df_results["BAG"].values
reg_bag = LinearRegression().fit(X_age, y_bag)
y_pred_bag = reg_bag.predict(X_age)

plt.plot(df_results["age"], y_pred_bag, color="darkorange", linestyle="--", label="Trendline")

# Labels and style
plt.title("BAG vs Chronological Age")
plt.xlabel("Age")
plt.ylabel("BAG (Brain Age Gap)")
plt.axhline(0, linestyle="--", color="gray")
plt.legend()
plt.tight_layout()
plt.show()

# ========================== Plot 2: cBAG vs Age ==========================
plt.figure(figsize=(8, 6))

# Scatter plot
sns.scatterplot(x="age", y="cBAG", data=df_results, color="steelblue", alpha=0.6, edgecolor="k")

# Trendline
y_cbag = df_results["cBAG"].values
reg_cbag = LinearRegression().fit(X_age, y_cbag)
y_pred_cbag = reg_cbag.predict(X_age)

plt.plot(df_results["age"], y_pred_cbag, color="navy", linestyle="--", label="Trendline")

# Labels and style
plt.title("Corrected BAG (cBAG) vs Chronological Age")
plt.xlabel("Age")
plt.ylabel("cBAG (Corrected Brain Age Gap)")
plt.axhline(0, linestyle="--", color="gray")
plt.legend()
plt.tight_layout()
plt.show()


#Metadata
import pandas as pd

# === Paths ===
SUBJECT_IDS_CSV = "/home/bas/Desktop/MyData/Columns/subject_ids.csv"
AGE_CSV = "/home/bas/Desktop/MyData/Columns/ages.csv"
METADATA_PATH = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx"

# === Load subject IDs and pad to 5 digits ===
subject_df = pd.read_csv(SUBJECT_IDS_CSV, header=None)
subject_ids = subject_df[0].astype(str).apply(lambda x: x.zfill(5)).tolist()

# === Load metadata ===
df_metadata = pd.read_excel(METADATA_PATH)

df_metadata["Risk"] = df_metadata["Risk"].fillna("NoRisk")


# Standardize MRI Exam ID column
df_metadata["MRI_Exam_fixed"] = (
    df_metadata["MRI_Exam"]
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# Drop fully empty rows and rows without MRI_Exam
df_metadata_cleaned = df_metadata.dropna(how="all")
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["MRI_Exam"])

# Filter metadata to only subjects we have columns for
df_matched = df_metadata_cleaned[df_metadata_cleaned["MRI_Exam_fixed"].isin(subject_ids)].copy()

# Optional: reset index
df_matched = df_matched.reset_index(drop=True)





# Merge predicted results with metadata
df_plot = pd.merge(df_results, df_matched, left_on="subject_id", right_on="MRI_Exam_fixed", how="left")



#Violin plots RISK
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")

# === BAG by Risk group ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="Risk", y="BAG", inner="point", palette="pastel")
plt.axhline(0, linestyle="--", color="gray")
plt.title("BAG by Risk Group")
plt.ylabel("Brain Age Gap (BAG)")
plt.xlabel("Risk Group")
plt.tight_layout()
plt.show()

# === cBAG by Risk group ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="Risk", y="cBAG", inner="point", palette="pastel")
plt.axhline(0, linestyle="--", color="gray")
plt.title("Corrected BAG (cBAG) by Risk Group")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.xlabel("Risk Group")
plt.tight_layout()
plt.show()



#VIOLIN GENOTYPE
import seaborn as sns
import matplotlib.pyplot as plt

# Set seaborn style
sns.set(style="whitegrid")

# Define correct APOE genotype order
genotype_order = ["APOE23", "APOE33", "APOE34", "APOE44"]

# === BAG by Genotype (ordered) ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="genotype", y="BAG", inner="point", palette="muted", order=genotype_order)
plt.axhline(0, linestyle="--", color="gray")
plt.title("BAG by Genotype")
plt.xlabel("Genotype")
plt.ylabel("Brain Age Gap (BAG)")
plt.tight_layout()
plt.show()

# === cBAG by Genotype (ordered) ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="genotype", y="cBAG", inner="point", palette="muted", order=genotype_order)
plt.axhline(0, linestyle="--", color="gray")
plt.title("Corrected BAG (cBAG) by Genotype")
plt.xlabel("Genotype")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.tight_layout()
plt.show()


#Violin genotype 4+ or 4-
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set(style="whitegrid")

# === BAG by APOE status ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="APOE", y="BAG", inner="point", palette="pastel")
plt.axhline(0, linestyle="--", color="gray")
plt.title("BAG by APOE4 Carrier Status")
plt.xlabel("APOE Status")
plt.ylabel("Brain Age Gap (BAG)")
plt.tight_layout()
plt.show()

# === cBAG by APOE status ===
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_plot, x="APOE", y="cBAG", inner="point", palette="pastel")
plt.axhline(0, linestyle="--", color="gray")
plt.title("Corrected BAG (cBAG) by APOE4 Carrier Status")
plt.xlabel("APOE Status")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.tight_layout()
plt.show()




