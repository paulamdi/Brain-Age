# === Clustering SHAP Contrastive Embeddings ===
# Discover latent groups (clusters) from SHAP-based embeddings using UMAP + KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap  # UMAP = Uniform Manifold Approximation and Projection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === 1. Load SHAP embeddings ===
df_embed = pd.read_csv("shap_embeddings.csv")  # Must include 'Subject_ID' + dim_0 ... dim_n

# === 2. Extract embedding matrix ===
embed_cols = [col for col in df_embed.columns if col.startswith("embed_")]
X = df_embed[embed_cols].values  # Shape: (n_subjects, embed_dim)

# Optional: Scale before UMAP (not mandatory but often helps)
X_scaled = StandardScaler().fit_transform(X)

# === 3. UMAP for 2D visualization ===
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Add UMAP coordinates to dataframe
df_embed["UMAP1"] = X_umap[:, 0]
df_embed["UMAP2"] = X_umap[:, 1]

# === 4. KMeans Clustering on original embeddings (not UMAP!) ===
k = 4  #  try 2, 3, 4, ... 
kmeans = KMeans(n_clusters=k, random_state=42)
df_embed["Cluster"] = kmeans.fit_predict(X_scaled)

# === 5. Plot UMAP with cluster assignments ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_embed, x="UMAP1", y="UMAP2",
    hue="Cluster", palette="Set2", s=60,
    edgecolor="k", linewidth=0.3
)
plt.title(f"UMAP of SHAP Embeddings with K-Means Clusters (k={k})")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === 6. (Optional) Save cluster assignments ===
df_embed[["Subject_ID", "UMAP1", "UMAP2","Cluster"]].to_csv("shap_embedding_clusters.csv", index=False)


