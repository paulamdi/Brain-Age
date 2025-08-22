# ===== SHAP Embeddings → (UMAP / t-SNE) → KMeans Clusters → (Optional) Merge + Associations =====
# - Unsupervised clustering on SHAP embeddings.
# - 2D visualization with UMAP and t-SNE (for plots only).
# - Pick k with silhouette (on original standardized embeddings).
# - Optionally merge with metadata XLS (MRI_Exam) and test Cluster associations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, fisher_exact

# ========= 0) Config =========
IN_EMB_CSV   = "FULL_FINE_TUNING_shap_embeddings.csv"  # must have 'Subject_ID' + 'embed_*'
IN_META_XLS  = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx"                         # your Excel with 'MRI_Exam' (optional merge step)
SEED         = 42
UMAP_PARAMS  = dict(n_neighbors=15, min_dist=0.1, random_state=SEED)
TSNE_PARAMS  = dict(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=SEED)

# ========= 1) Load embeddings =========
df = pd.read_csv(IN_EMB_CSV)
embed_cols = [c for c in df.columns if c.startswith("embed_")]
assert len(embed_cols) > 0, "No columns found with prefix 'embed_*' in the embeddings CSV."
X = df[embed_cols].values

# ========= 2) Standardize (recommended) =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========= 3) 2D projections (visualization only) =========
# UMAP
umap_model = umap.UMAP(**UMAP_PARAMS)
X_umap = umap_model.fit_transform(X_scaled)
df["UMAP1"], df["UMAP2"] = X_umap[:, 0], X_umap[:, 1]

# t-SNE
tsne_model = TSNE(**TSNE_PARAMS)
X_tsne = tsne_model.fit_transform(X_scaled)
df["TSNE1"], df["TSNE2"] = X_tsne[:, 0], X_tsne[:, 1]

# ========= 4) Choose k via silhouette (on X_scaled, not UMAP/TSNE) =========
candidate_k = list(range(2, 9))
sil_scores = []
for k in candidate_k:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    labels_k = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels_k))

best_k = candidate_k[int(np.argmax(sil_scores))]
print(f"[INFO] Best k by silhouette: k={best_k} (score={max(sil_scores):.3f})")

# ========= 5) Final KMeans on X_scaled =========
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=SEED)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ========= 6) Plots (saved) =========
# UMAP
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="Cluster",
                palette="Set2", s=60, edgecolor="k", linewidth=0.3)
plt.title(f"UMAP of SHAP Embeddings + KMeans (k={best_k})")
plt.tight_layout()
plt.savefig("shap_umap_kmeans.png", dpi=300)

plt.close()

# t-SNE
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="Cluster",
                palette="Set2", s=60, edgecolor="k", linewidth=0.3)
plt.title(f"t-SNE of SHAP Embeddings + KMeans (k={best_k})")
plt.tight_layout()
plt.savefig("shap_tsne_kmeans.png", dpi=300)

plt.close()

# Silhouette vs k
plt.figure(figsize=(6, 4))
plt.plot(candidate_k, sil_scores, marker="o")
plt.xticks(candidate_k)
plt.xlabel("k (clusters)")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs k (on original embeddings)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("silhouette_vs_k.png", dpi=300)

plt.close()

# ========= 7) Save cluster assignments + projections =========
cols_out = ["Subject_ID", "Cluster", "UMAP1", "UMAP2", "TSNE1", "TSNE2"] + embed_cols
df[cols_out].to_csv("shap_embedding_clusters_with_projections.csv", index=False)
print("[INFO] Saved: shap_embedding_clusters_with_projections.csv")

# ========= 8) (Optional) Merge with metadata XLS and run association tests =========
try:
    df_meta = pd.read_excel(IN_META_XLS)  # must contain 'MRI_Exam'

    # --- normaliza ambas claves a STRING ---
    # Embeddings CSV: si Subject_ID viene numérico, pásalo a str y zfill(5)
    if np.issubdtype(df["Subject_ID"].dtype, np.number):
        df["Subject_ID"] = df["Subject_ID"].astype(int).astype(str).str.zfill(5)
    else:
        df["Subject_ID"] = df["Subject_ID"].astype(str).str.zfill(5)

    # Metadata XLS: crea Subject_ID desde MRI_Exam en str zfill(5)
    if "MRI_Exam" not in df_meta.columns:
        raise KeyError("Column 'MRI_Exam' not found in metadata XLS.")

    if np.issubdtype(df_meta["MRI_Exam"].dtype, np.number):
        df_meta["Subject_ID"] = df_meta["MRI_Exam"].astype(int).astype(str).str.zfill(5)
    else:
        # por si viene como '00123' string ya
        df_meta["Subject_ID"] = df_meta["MRI_Exam"].astype(str).str.zfill(5)

    # --- merge ---
    df_merged = pd.merge(df, df_meta, on="Subject_ID", how="inner")
    df_merged.to_csv("shap_embedding_clusters_with_metadata.csv", index=False)
    print(f"[INFO] Saved merged file: shap_embedding_clusters_with_metadata.csv (shape={df_merged.shape})")

    # (opcional) tabla de tamaños por cluster
    print(df_merged["Cluster"].value_counts().sort_index())

    # Association helper
    def assoc_table_and_test(df_in, cluster_col, label_col):
        if label_col not in df_in.columns:
            print(f"[WARN] Column not found: {label_col}")
            return
        tab = pd.crosstab(df_in[cluster_col], df_in[label_col])
        print(f"\n[Contingency] {cluster_col} vs {label_col}\n", tab)
        try:
            chi2, p, dof, exp = chi2_contingency(tab)
            print(f"[Chi^2] chi2={chi2:.3f}, dof={dof}, p={p:.4g}")
        except ValueError:
            print("[Chi^2] Not applicable (singular table). Trying Fisher...")
            if tab.shape == (2, 2):
                _, p_f = fisher_exact(tab.values)
                print(f"[Fisher 2x2] p={p_f:.4g}")
            else:
                print("[Fisher] Not 2x2.")

    for label in ["Risk", "APOE", "sex", "genotype"]:
        assoc_table_and_test(df_merged, "Cluster", label)

except FileNotFoundError:
    print(f"[WARN] Metadata file not found: {IN_META_XLS} — skipping merge and association tests.")
except Exception as e:
    print(f"[WARN] Skipped merge/associations due to error: {e}")

