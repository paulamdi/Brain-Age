# ==============================================================
# GLASS BRAIN – AD-DECODE   |  Top-10 intra-Left, intra-Right, Inter
# ==============================================================

import os, numpy as np, pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from nilearn import plotting

# ------------------------------------------------------------------
# 1)  CSV SHAP (AD-DECODE)
# ------------------------------------------------------------------
shap_dir = ("/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/"
            "BAG/ADDECODE Saving figures/shap_outputs_addecode")
all_shap_dfs = []

for fname in os.listdir(shap_dir):
    if fname.startswith("edge_shap_subject_") and fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        all_shap_dfs.append(df)

shap_df = pd.concat(all_shap_dfs, ignore_index=True)

# ------------------------------------------------------------------
# 2) Replace index for regions names 
# ------------------------------------------------------------------
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

shap_df["Region_1"] = shap_df["Node_i"].apply(lambda x: region_names[int(x)])
shap_df["Region_2"] = shap_df["Node_j"].apply(lambda x: region_names[int(x)])

# ------------------------------------------------------------------
# 3) Group and clasify connections
# ------------------------------------------------------------------
# Nota: la columna de interés es `SHAP_val` (no `SHAP_value`)
grouped = shap_df.groupby(["Region_1", "Region_2"])["SHAP_val"].mean().reset_index()

def conn_type(r1, r2):
    if r1.startswith("Left")  and r2.startswith("Left"):  return "Intra-Left"
    if r1.startswith("Right") and r2.startswith("Right"): return "Intra-Right"
    return "Inter"

grouped["Type"] = grouped.apply(lambda r: conn_type(r["Region_1"], r["Region_2"]), axis=1)

top10_left  = grouped[grouped["Type"] == "Intra-Left"] .nlargest(10, "SHAP_val")
top10_right = grouped[grouped["Type"] == "Intra-Right"].nlargest(10, "SHAP_val")
top10_inter = grouped[grouped["Type"] == "Inter"]      .nlargest(10, "SHAP_val")
top_combined = pd.concat([top10_left, top10_right, top10_inter], ignore_index=True)

# ------------------------------------------------------------------
# 4)  CENTROIDS from the  ATLAS 
# ------------------------------------------------------------------
img = nib.load("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_labels.nii.gz")
data, affine = img.get_fdata(), img.affine

labels = np.unique(data)[1:]  # removes 0
centroids = [nib.affines.apply_affine(affine, np.argwhere(data == l).mean(0)) for l in labels]
centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"]); centroid_df["Label"] = labels

lookup = pd.read_excel("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_lookup.xlsx")
final_df = pd.merge(centroid_df, lookup, left_on="Label", right_on="Index")
name2coord = {row["Structure"]: [row["X"], row["Y"], row["Z"]] for _, row in final_df.iterrows()}

# ------------------------------------------------------------------
# 5) plot function
# ------------------------------------------------------------------
def plot_glass_brain(top_df, title, save_path):
    regs = list(set(top_df["Region_1"]) | set(top_df["Region_2"]))
    reg2idx = {r:i for i, r in enumerate(regs)}
    coords  = [name2coord[r] for r in regs]

    n = len(regs)
    mat = np.zeros((n, n))
    for _, row in top_df.iterrows():
        i, j = reg2idx[row["Region_1"]], reg2idx[row["Region_2"]]
        mat[i, j] = row["SHAP_val"]; mat[j, i] = row["SHAP_val"]

    cmap_nodes = cm.get_cmap('tab10', n)
    colors = [cmap_nodes(i) for i in range(n)]

    fig = plt.figure(figsize=(12, 6))
    ax_brain  = fig.add_axes([0.05, 0.05, 0.7, 0.9])
    ax_legend = fig.add_axes([0.78, 0.2, 0.2, 0.6])
    plotting.plot_connectome(mat, coords, edge_threshold="0%",
                             node_color=colors, node_size=100,
                             edge_cmap=plt.cm.Reds, axes=ax_brain)

    fig.suptitle(title, fontsize=20, y=0.96)
    ax_legend.axis('off')
    patches = [mpatches.Patch(color=colors[i], label=f"{i:02d} → {r}") for i, r in enumerate(regs)]
    ax_legend.legend(handles=patches, loc='center left', fontsize=11, frameon=False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✔ Saved: {save_path}")

# ------------------------------------------------------------------
# 6) save figures
# ------------------------------------------------------------------
out_dir = "figs_glass_brain_dti_addecode"
plot_glass_brain(top10_left,  "AD-DECODE Top-10 Intra-Left DTI SHAP",   f"{out_dir}/glass_intra_left.png")
plot_glass_brain(top10_right, "AD-DECODE Top-10 Intra-Right DTI SHAP",  f"{out_dir}/glass_intra_right.png")
plot_glass_brain(top10_inter, "AD-DECODE Top-10 Interhemispheric DTI SHAP", f"{out_dir}/glass_inter.png")
plot_glass_brain(top_combined, "AD-DECODE Top-30 DTI SHAP (All Types)", f"{out_dir}/glass_all_types.png")
