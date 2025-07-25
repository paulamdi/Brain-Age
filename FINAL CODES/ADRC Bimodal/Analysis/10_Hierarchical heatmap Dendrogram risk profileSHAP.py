#Hierarchical clustering heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import os

# === Load SHAP and metadata CSV ===
csv_path = r"C:\Users\Paula\OneDrive\Escritorio\INTERNSHIP_PAULA\Paula DTI_fMRI Codes\ADRC\BEST\shap_cluster_analysis\merged_shap_clusters.csv"
output_dir = r"C:\Users\Paula\OneDrive\Escritorio\INTERNSHIP_PAULA\Paula DTI_fMRI Codes\ADRC\BEST\shap_cluster_analysis1"
os.makedirs(output_dir, exist_ok=True)


selected_cols = [
    "Subject_ID", "Sex", "Genotype",
    "DTI_Clustering", "DTI_PathLen", "DTI_GlobalEff", "DTI_LocalEff",
    "fMRI_Clustering", "fMRI_PathLen", "fMRI_GlobalEff", "fMRI_LocalEff",
    "AB40_x", "AB42_x", "AB_ratio_x", "TTAU_x", "PTAU181_x"
]

df = pd.read_csv(csv_path, usecols=selected_cols)

# === Clean and extract metadata ===
df["Sex"] = df["Sex"].astype(str).str.upper().str.strip()
df["Sex"] = df["Sex"].replace({"1": "M", "2": "F", "MALE": "M", "FEMALE": "F"})
subject_ids = df["Subject_ID"]
metadata_df = df[["Subject_ID"]]

# === Extract SHAP matrix and z-score ===
shap_matrix = df.select_dtypes(include=[np.number]).fillna(0)
X_z = pd.DataFrame(StandardScaler().fit_transform(shap_matrix), index=subject_ids, columns=shap_matrix.columns)

# === KMeans for cluster color bar (k=3) ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_z)

# === Row color bar: Cluster + Sex + Risk + APOE ===
palette_cluster = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
palette_sex     = {"F": "#9467bd", "M": "#8c564b"}
palette_risk    = {"NoRisk": "#17becf", "MCI": "#e377c2", "AD": "#d62728"}
palette_apoe    = {"E4+": "#7f7f7f", "E4-": "#bcbd22"}

row_colors = pd.DataFrame(index=subject_ids)
row_colors["Cluster"] = pd.Series(cluster_labels, index=subject_ids).map(palette_cluster)


# === Column color bar: feature categories ===
def feature_group(f):
    if f in ["Age", "BMI", "Sex"]: return "Demographic"
    elif "EFF" in f or "Clust" in f or "Path" in f: return "GraphMetric"
    elif f.startswith("PC"): return "PCA_gene"
    elif f in ["APOE", "AB_ratio", "TTAU", "PTAU181", "GFAP", "NFL"]: return "Biomarker"
    else: return "Other"

col_groups = X_z.columns.to_series().apply(feature_group)
palette_feat = {
    "Demographic": "#e377c2", "GraphMetric": "#17becf",
    "PCA_gene": "#ff7f0e", "Biomarker": "#bcbd22", "Other": "#7f7f7f"
}
col_colors = col_groups.map(palette_feat)

# === Bivariate clustermap ===
g = sns.clustermap(
    X_z,
    cmap="vlag",
    center=0,
    vmin=-3, vmax=3,
    row_cluster=True,
    col_cluster=True,
    row_colors=row_colors,
    col_colors=col_colors,
    dendrogram_ratio=(.15, .15),
    figsize=(14, 10),
    xticklabels=True,
    yticklabels=True,
    cbar_kws={"label": "Z-scored SHAP value"}
)


# === Adjust colorbar position (move upward) ===
# get current colorbar position
cbar_ax = g.cax
pos = cbar_ax.get_position()

# move it upward by a small amount (e.g., 0.02 in figure coordinates)
cbar_ax.set_position([
    pos.x0,         # left (no change)
    pos.y0 + 0.08,  # bottom → shift upward
    pos.width,      # same width
    pos.height      # same height
])


# === Rotate feature labels for readability ===
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=6)

# === Legends for color bars ===
for title, palette in zip(
    ["Cluster"],
    [palette_cluster]
):
    handles = [plt.Line2D([0], [0], marker='s', color=color, label=lab, linestyle='', markersize=8)
               for lab, color in palette.items()]
    g.ax_row_dendrogram.legend(handles=handles, title=title, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

# Feature group legend
feat_handles = [plt.Line2D([0], [0], marker='s', color=color, label=lab, linestyle='', markersize=8)
                for lab, color in palette_feat.items()]
# --------------------------------------------------------------
# Build the list of feature-group colour chips
# --------------------------------------------------------------
feat_handles = [
    plt.Line2D([0], [0], marker='s', color=c, label=l,
               linestyle='', markersize=8)
    for l, c in palette_feat.items()
    if l in col_groups.values                        # keep only groups present
]

# --------------------------------------------------------------
# Add legend *only if* we have at least one handle
# NOTE: use the keyword  handles= …  so a single handle is OK
# --------------------------------------------------------------
if feat_handles:                                      # list not empty
    g.ax_col_dendrogram.legend(
        handles      = feat_handles,                  # <- keyword!
        title        = "Feature Group",
        bbox_to_anchor = (0.5, 1.1),
        loc          = "lower center",
        ncol         = max(1, len(feat_handles) // 2 + 1),
        frameon      = False
    )

# === Save and show ===
plt.suptitle("Bivariate Hierarchical Clustering of SHAP Profiles", y=1.1)
plt.savefig(os.path.join(output_dir, "shap_clustermap_bivariate_colored.png"), dpi=300, bbox_inches="tight")
plt.show()




# --------------------------------------------------
# Build row_colors WITHOUT the cluster bar
# --------------------------------------------------
row_colors_nocluster = pd.DataFrame(index=subject_ids)



# --------------------------------------------------
# Clustermap no  “Cluster”
# --------------------------------------------------
g2 = sns.clustermap(
    X_z,
    cmap="vlag",
    center=0,
    vmin=-3, vmax=3,
    row_cluster=True,
    col_cluster=True,
    row_colors=None,         # ← no pass an empty DataFrame
    col_colors=col_colors,   # ← this can stay
    dendrogram_ratio=(.15, .15),
    figsize=(14, 10),
    xticklabels=True,
    yticklabels=True,
    cbar_kws={"label": "Z-scored SHAP value"}
)

# === Adjust colorbar position (move upward) ===
# get current colorbar position
cbar_ax = g2.cax
pos = cbar_ax.get_position()

# move it upward by a small amount (e.g., 0.02 in figure coordinates)
cbar_ax.set_position([
    pos.x0,         # left (no change)
    pos.y0 + 0.08,  # bottom → shift upward
    pos.width,      # same width
    pos.height      # same height
])



plt.setp(g2.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=6)

# (Opcional) leyendas solo de las anotaciones que sigan presentes
# ...

plt.suptitle("SHAP Clustermap (no left Cluster bar)", y=1.1)
plt.savefig(os.path.join(output_dir, "shap_clustermap_noClusterBar.png"),
            dpi=300, bbox_inches="tight")
plt.show()




