# GLASS BRAIN FMRI left right inter

import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nilearn import plotting
import matplotlib.cm as cm

# === Load SHAP values from all fMRI subjects ===
shap_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.3 shap edges fmri/shap_outputs_fmri"
all_shap_dfs = []

for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_fmri_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        all_shap_dfs.append(df)

shap_df = pd.concat(all_shap_dfs, ignore_index=True)

# === Brain region names ===
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
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts",
    "ctx-rh-caudalanteriorcingulate", "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal",
    "ctx-rh-fusiform", "ctx-rh-inferiorparietal", "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate",
    "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal", "ctx-rh-lingual", "ctx-rh-medialorbitofrontal",
    "ctx-rh-middletemporal", "ctx-rh-parahippocampal", "ctx-rh-paracentral", "ctx-rh-parsopercularis",
    "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis", "ctx-rh-pericalcarine", "ctx-rh-postcentral",
    "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus", "ctx-rh-rostralanteriorcingulate",
    "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal", "ctx-rh-superiorparietal", "ctx-rh-superiortemporal",
    "ctx-rh-supramarginal", "ctx-rh-frontalpole", "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

shap_df["Region_1"] = shap_df["Node_i"].apply(lambda x: region_names[int(x)])
shap_df["Region_2"] = shap_df["Node_j"].apply(lambda x: region_names[int(x)])

# === Group by edge and average SHAP ===
grouped = shap_df.groupby(["Region_1", "Region_2"])["SHAP_value"].mean().reset_index()

# === Classify edge type ===
def classify_connection(r1, r2):
    if r1.startswith("Left") and r2.startswith("Left"):
        return "Intra-Left"
    elif r1.startswith("Right") and r2.startswith("Right"):
        return "Intra-Right"
    else:
        return "Inter"

grouped["Type"] = grouped.apply(lambda row: classify_connection(row["Region_1"], row["Region_2"]), axis=1)

# === Top 10 edges per type ===
top10_left = grouped[grouped["Type"] == "Intra-Left"].sort_values(by="SHAP_value", ascending=False).head(10)
top10_right = grouped[grouped["Type"] == "Intra-Right"].sort_values(by="SHAP_value", ascending=False).head(10)
top10_inter = grouped[grouped["Type"] == "Inter"].sort_values(by="SHAP_value", ascending=False).head(10)

# === Combine ===
top_combined = pd.concat([top10_left, top10_right, top10_inter], ignore_index=True)
top_combined["EdgeType"] = (
    ["Intra-Left"] * len(top10_left) +
    ["Intra-Right"] * len(top10_right) +
    ["Inter"] * len(top10_inter)
)

# === Load centroids ===
img = nib.load("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_labels.nii.gz")
data = img.get_fdata()
affine = img.affine

region_labels = np.unique(data)
region_labels = region_labels[region_labels != 0]

centroids = []
for label in region_labels:
    mask = data == label
    coords = np.argwhere(mask)
    center_voxel = coords.mean(axis=0)
    center_mni = nib.affines.apply_affine(affine, center_voxel)
    centroids.append(center_mni)

centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
centroid_df["Label"] = region_labels.astype(int)

lookup = pd.read_excel("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_lookup.xlsx")
final_df = pd.merge(centroid_df, lookup, left_on="Label", right_on="Index")
region_name_to_coords = {
    row["Structure"]: [row["X"], row["Y"], row["Z"]] for _, row in final_df.iterrows()
}

# === Plotting function ===
def plot_glass_brain(top_df, title, save_name):
    regions_involved = list(set(top_df["Region_1"]) | set(top_df["Region_2"]))
    region_to_index = {region: idx for idx, region in enumerate(regions_involved)}
    coords = [region_name_to_coords[region] for region in regions_involved]

    con_matrix = np.zeros((len(regions_involved), len(regions_involved)))
    for _, row in top_df.iterrows():
        i = region_to_index[row["Region_1"]]
        j = region_to_index[row["Region_2"]]
        con_matrix[i, j] = row["SHAP_value"]
        con_matrix[j, i] = row["SHAP_value"]

    cmap = cm.get_cmap('tab10', len(regions_involved))
    node_colors = [cmap(i) for i in range(len(regions_involved))]

    fig = plt.figure(figsize=(12, 6))
    ax_brain = fig.add_axes([0.05, 0.05, 0.7, 0.9])
    ax_legend = fig.add_axes([0.77, 0.2, 0.2, 0.6])

    plotting.plot_connectome(
        con_matrix,
        coords,
        edge_threshold="0%",
        node_color=node_colors,
        node_size=100,
        edge_cmap=plt.cm.Reds,
        axes=ax_brain,
        title=None
    )

    fig.suptitle(title, fontsize=20, y=0.95)

    ax_legend.axis('off')
    legend_patches = [
        mpatches.Patch(color=node_colors[i], label=f"{i:02d} → {region}")
        for i, region in enumerate(regions_involved)
    ]
    ax_legend.legend(
        handles=legend_patches,
        loc='center left',
        fontsize=12,
        frameon=False
    )

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
# === Create output folder ===
output_dir = "figs_glass_brain_fmri"
os.makedirs(output_dir, exist_ok=True)

# === Plot all and save in folder ===
plot_glass_brain(top10_left, "Top 10 Intra-Left fMRI SHAP Connections", os.path.join(output_dir, "glass_brain_fmri_intra_left.png"))
plot_glass_brain(top10_right, "Top 10 Intra-Right fMRI SHAP Connections", os.path.join(output_dir, "glass_brain_fmri_intra_right.png"))
plot_glass_brain(top10_inter, "Top 10 Interhemispheric fMRI SHAP Connections", os.path.join(output_dir, "glass_brain_fmri_inter.png"))
plot_glass_brain(top_combined, "Top 30 fMRI SHAP Connections (All Types)", os.path.join(output_dir, "glass_brain_fmri_all.png"))
