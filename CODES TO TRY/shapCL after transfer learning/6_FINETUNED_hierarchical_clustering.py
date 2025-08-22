#HIERARCHICAL CLUSTERING FROM CL shap embeddings after full fine tuning


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load embeddings ===
df = pd.read_csv("FULL_FINE_TUNING_shap_embeddings.csv")

# === Step 2: Extract embedding columns ===
embed_cols = [col for col in df.columns if col.startswith("embed_")]
embedding_matrix = df[embed_cols]
embedding_matrix.index = df["Subject_ID"]

# === Step 3: Clustered heatmap ===
g = sns.clustermap(
    embedding_matrix,
    cmap="coolwarm",
    row_cluster=True,
    col_cluster=True,
    yticklabels=True,          # Show subject IDs on y-axis
    figsize=(14, 10),
    cbar_kws={"label": "Embedding Value"}
)

# === Step 4: Save the figure ===
g.savefig("FULL_FINE_TUNING_CL_shap_embeddings_clustermap.png", dpi=300, bbox_inches="tight")

print(" Clustermap saved as shap_embeddings_clustermap.png")
