
# =========================================================================================
# shap_clustering.py

#   1) Loads the merged SHAP coefficient matrix stored in `merged_shap_clusters.csv`
#   2) Drops the subject-identifier column
#   3) (Optionally) z-scores the SHAP values
#   4) Reduces dimensionality to 2-D with UMAP for visualization
#   5) Finds the optimal number of clusters via silhouette score (K-means, k = 2-10)
#   6) Fits K-means with that k and assigns a cluster label to every subject
#   7) Saves a scatter plot (`shap_clusters_umap.png`) and a CSV with labels

# =========================================================================================

# ============================
# Cluster subjects using SHAP
# ============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import umap.umap_ as umap
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")       





# === Paths ===
csv_path = r"C:\Users\Paula\OneDrive\Escritorio\INTERNSHIP_PAULA\Paula DTI_fMRI Codes\ADRC\BEST\shap_cluster_analysis\merged_shap_clusters.csv"
output_dir = "shap_cluster_analysis2"
os.makedirs(output_dir, exist_ok=True)

# === Load and preprocess data ===
df = pd.read_csv(csv_path)
subject_ids = df["Subject_ID"]  # store for later
shap_matrix = df.select_dtypes(include=[np.number])  # drop ID and keep numeric only

# === Handle missing values ===
shap_matrix = shap_matrix.fillna(0)  # or use .dropna() to remove subjects with NaNs

# === Z-score normalization ===
scaler = StandardScaler()
shap_matrix = scaler.fit_transform(shap_matrix)

# === UMAP dimensionality reduction ===
reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, metric="euclidean")
embedding = reducer.fit_transform(shap_matrix)

# === Find optimal k using silhouette score ===
best_k = 2
best_score = -1

print("[INFO] Finding optimal k (silhouette score)...")
for k in range(2, 11):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_tmp = kmeans_tmp.fit_predict(shap_matrix)
    score_tmp = silhouette_score(shap_matrix, labels_tmp)
    print(f"  k = {k:<2d}  -> silhouette = {score_tmp:.4f}")
    if score_tmp > best_score:
        best_k, best_score = k, score_tmp

print(f"[INFO] Selected k = {best_k} (silhouette = {best_score:.4f})")

# === Final KMeans fit ===
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(shap_matrix)

# === Add labels to DataFrame and save ===
df["Cluster"] = cluster_labels
df.to_csv(os.path.join(output_dir, "shap_clusters_with_labels.csv"), index=False)
print("[INFO] Clustered CSV saved.")

# === Plot UMAP with cluster labels ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap="tab10", s=60, edgecolor="k")
plt.title("Subject Clusters (UMAP + KMeans)", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_clusters_umap.png"), dpi=300)
plt.show()
print("[INFO] UMAP plot saved.")


# === Manual color bin assignment ===
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

plt.figure(figsize=(8, 6))
for cluster_id in range(3):
    idx = cluster_labels == cluster_id
    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        c=colors[cluster_id],
        label=f"Cluster {cluster_id}",
        s=60,
        edgecolor="k"
    )

plt.title("Subject Clusters (UMAP + KMeans, k=3)", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster", loc="best")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_clusters_umap.png"), dpi=300)
plt.show()
print("[INFO] UMAP cluster plot saved.")













import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import os



# --- load & z-score ---
df          = pd.read_csv(csv_path)
subject_ids = df["Subject_ID"]
X           = df.select_dtypes(include=[np.number]).fillna(0)

X_z = pd.DataFrame(
    StandardScaler().fit_transform(X),
    index = subject_ids,
    columns = X.columns
)

# --- K-means (k=3) just to colour the rows ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_z)

# colour map for three clusters
row_palette = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
row_colors  = pd.Series(cluster_labels, index=X_z.index).map(dict(zip(range(3), row_palette.colors)))





# You already have this from earlier KMeans or clustering
df_clusters = pd.DataFrame({
    "Subject_ID": X_z.index,
    "Cluster": cluster_labels
})

# Merge with metadata
df_meta = df[["Subject_ID", "NORMCOG", "SUBJECT_SEX", "APOE"]]
df_merged = pd.merge(df_clusters, df_meta, on="Subject_ID")



# Count metadata distribution per cluster
counts_risk = pd.crosstab(df_merged["Cluster"], df_merged["NORMCOG"])
counts_sex  = pd.crosstab(df_merged["Cluster"], df_merged["SUBJECT_SEX"])
counts_apoe = pd.crosstab(df_merged["Cluster"], df_merged["APOE"])

print("=== Risk per Cluster ===")
print(counts_risk, "\n")

print("=== Sex per Cluster ===")
print(counts_sex, "\n")

print("=== APOE Status per Cluster ===")
print(counts_apoe)







# Distribution of APOE status per cluster
apoe_counts = pd.crosstab(df_merged["APOE"], df_merged["Cluster"])

# Distribution of Sex per cluster
sex_counts = pd.crosstab(df_merged["SUBJECT_SEX"], df_merged["Cluster"])



# === Define consistent cluster colors ===
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Cluster 0, 1, 2

# === Risk Group Composition ===
counts_risk.T.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors)
plt.title("Risk Group Composition per SHAP Cluster")
plt.xlabel("Risk Group")
plt.ylabel("Number of Subjects")
plt.legend(title="Cluster", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_risk_cluster.png"), dpi=300)
plt.close()

# === APOE Status Composition ===
apoe_counts.plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)
plt.title("APOE Status Composition per SHAP Cluster")
plt.xlabel("APOE Status")
plt.ylabel("Number of Subjects")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_apoe_cluster.png"), dpi=300)
plt.close()

# === Sex Composition ===
sex_counts.plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)
plt.title("Sex Composition per SHAP Cluster")
plt.xlabel("Sex")
plt.ylabel("Number of Subjects")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_sex_cluster.png"), dpi=300)
plt.close()







# Bar plot where X-axis is Cluster, and each bar shows stacked categories



# === Cluster-wise composition of Risk groups ===
counts_risk.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors)
plt.title("SHAP Cluster Composition by Risk Group")
plt.xlabel("SHAP Cluster")
plt.ylabel("Number of Subjects")
plt.legend(title="Risk Group", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_cluster_riskgroup.png"), dpi=300)
plt.close()




# APOE composition by SHAP cluster (cluster on X-axis)


# Build cross-tab for APOE composition
apoe_counts = pd.crosstab(df_merged["Cluster"], df_merged["APOE"])

# Generate one color per APOE genotype dynamically
apoe_palette = sns.color_palette("tab20", n_colors=apoe_counts.shape[1])

# Plot with dynamic palette
apoe_counts.plot(kind="bar",
                 stacked=True,
                 figsize=(8, 5),
                 color=apoe_palette)
plt.title("SHAP Cluster Composition by APOE Status")
plt.xlabel("SHAP Cluster")
plt.ylabel("Number of Subjects")
plt.legend(title="APOE Status", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_cluster_apoe.png"), dpi=300)
plt.close()






# === Cluster-wise composition of Sex ===
sex_counts.plot(kind='bar', stacked=True, figsize=(8, 5), color=colors)
plt.title("SHAP Cluster Composition by Sex")
plt.xlabel("SHAP Cluster")
plt.ylabel("Number of Subjects")
plt.legend(title="Sex", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_cluster_sex.png"), dpi=300)
plt.close()



# Sex composition by SHAP cluster (cluster on X-axis)


# 1) Crosstab: Cluster as rows, Sex as columns  ⬅️ key change
sex_counts = pd.crosstab(df_merged["Cluster"], df_merged["SUBJECT_SEX"])

# 2) Stacked bar plot
sex_counts.plot(kind="bar",
                stacked=True,
                figsize=(8, 5),
                color=colors)                 # re-use cluster colour palette
plt.title("SHAP Cluster Composition by Sex")
plt.xlabel("SHAP Cluster")
plt.ylabel("Number of Subjects")
plt.legend(title="Sex", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bar_cluster_sex.png"), dpi=300)
plt.close()
