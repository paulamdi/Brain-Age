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

import pickle

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




with open("graph_data_list_addecode.pkl", "rb") as f:
    graph_data_list_addecode = pickle.load(f)

with open("age_bins_addecode.pkl", "rb") as f:
    age_bins = pickle.load(f)

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
_, test_idx = list(skf.split(graph_data_list_addecode, age_bins))[0]
test_data = [graph_data_list_addecode[i] for i in test_idx]
test_ages = [data.y.item() for data in test_data]

i_young = np.argmin(test_ages)
i_old = np.argmax(test_ages)
i_mid = np.abs(test_ages - np.median(test_ages)).argmin()
selected_subjects = [i_young, i_mid, i_old]









model = BrainAgeGATv2(global_feat_dim=7).to(device)
model.load_state_dict(torch.load("finetuned2_fold_1_rep_6.pt", map_location=device))
model.eval()

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


def extract_attention_matrices(data):
    data = data.to(device)
    x = model.node_embed(data.x)
    x1, (ei1, att1) = model.gnn1(x, data.edge_index, return_attention_weights=True)
    x1 = F.relu(model.bn1(x1))
    x2, (ei2, att2) = model.gnn2(x1, data.edge_index, return_attention_weights=True)
    x2 = F.relu(model.bn2(x2 + x1))
    x3, (ei3, att3) = model.gnn3(x2, data.edge_index, return_attention_weights=True)
    x3 = F.relu(model.bn3(x3 + x2))
    x4, (ei4, att4) = model.gnn4(x3, data.edge_index, return_attention_weights=True)
    x4 = F.relu(model.bn4(x4 + x3))

    def to_df(ei, att):
        ei = ei.cpu().numpy().T
        att = att.detach().cpu().numpy().mean(axis=1)
        df = pd.DataFrame(ei, columns=["source_node", "target_node"])
        df["attention_score"] = att
        df = df[df["source_node"] != df["target_node"]].copy()
        df["region_source"] = df["source_node"].apply(lambda i: region_names[i])
        df["region_target"] = df["target_node"].apply(lambda i: region_names[i])
        return df.sort_values("attention_score", ascending=False).reset_index(drop=True)

    return [to_df(*x) for x in [(ei1, att1), (ei2, att2), (ei3, att3), (ei4, att4)]]





attention_dfs = []
attention_matrices = []
subject_labels = []

for idx in selected_subjects:
    subj = test_data[idx]
    mats_df = extract_attention_matrices(subj)
    attention_dfs.append(mats_df)
    subject_labels.append(f"Age {subj.y.item():.1f}")
    for i, df in enumerate(mats_df):
        df.to_csv(f"subject_age_{int(subj.y.item())}_layer_gnn{i+1}.csv", index=False)
    mats = []
    for df in mats_df:
        mat = np.zeros((84, 84))
        for _, row in df.iterrows():
            i, j, score = int(row["source_node"]), int(row["target_node"]), row["attention_score"]
            mat[i, j] = score
            mat[j, i] = score
        if mat.max() > 0:
            mat /= mat.max()
        mats.append(mat)
    attention_matrices.append(mats)



fig, axs = plt.subplots(3, 4, figsize=(20, 12))
layer_titles = ["gnn1", "gnn2", "gnn3", "gnn4"]

for row in range(3):
    for col in range(4):
        sns.heatmap(attention_matrices[row][col], ax=axs[row, col], square=True, cmap="viridis", cbar=True)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        if row == 0:
            axs[row, col].set_title(layer_titles[col], fontsize=14)
        if col == 0:
            axs[row, col].set_ylabel(subject_labels[row], fontsize=14)

plt.suptitle("Attention Evolution Across GNN Layers (Subjects: Young, Median, Old)", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("attention_heatmap_3subjects.png", dpi=300)
plt.show()





for subj_idx, subj_dfs in enumerate(attention_dfs):
    print(f"\n Subject {subj_idx + 1} ({subject_labels[subj_idx]}):")
    for layer_idx, df in enumerate(subj_dfs):
        print(f"   Layer gnn{layer_idx + 1}:")
        top5 = df.head(5)
        for _, row in top5.iterrows():
            print(f"     → {row['region_source']} → {row['region_target']} | Score: {row['attention_score']:.4f}")




# === Define region names (index 0 to 83) ===
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

# === Map indices to hemispheres ===
left_indices = [i for i, name in enumerate(region_names) if "Left" in name or "lh" in name]
right_indices = [i for i, name in enumerate(region_names) if "Right" in name or "rh" in name]

# === Compute attention sums from matrix ===
# Replace 'mat' with the name of your 84x84 attention matrix
total_attention_per_node = mat.sum(axis=1)

left_attention = total_attention_per_node[left_indices].sum()
right_attention = total_attention_per_node[right_indices].sum()

print(f"\n Attention distribution by hemisphere:")
print(f"  → Left Hemisphere:  {left_attention:.4f}")
print(f"  → Right Hemisphere: {right_attention:.4f}")
