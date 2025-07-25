import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load the SHAP embeddings with predicted risk ===
df = pd.read_csv("shap_embeddings_with_riskprob.csv")


#Heatmap ordered by risk

# === Step 2: Sort subjects by estimated risk probability ===
#df_sorted = df.sort_values(by="RiskProb", ascending=True).reset_index(drop=True)


# === Step 3: Extract only the embedding columns ===
#embed_cols = [col for col in df.columns if col.startswith("embed_")]
#embedding_matrix = df_sorted[embed_cols].values  # Shape: (n_subjects, embed_dim)

# === Step 4: Create the heatmap ===
#plt.figure(figsize=(12, 8))
#sns.heatmap(
#    embedding_matrix,
#    cmap="coolwarm",
#    yticklabels=False,
#    cbar_kws={"label": "Embedding Value"},
#    xticklabels=embed_cols
#)

#plt.title("Subject-Level SHAP Embedding Profiles (Sorted by Predicted Risk)")
#plt.xlabel("Embedding Dimension")
#plt.ylabel("Subjects (Ordered by Risk Probability)")
#plt.tight_layout()
#plt.show()




#heatmap ordered clustering

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === Step 1: Extract embeddings ===
embed_cols = [col for col in df.columns if col.startswith("embed_")]
embedding_matrix = df[embed_cols]
embedding_matrix.index = df["Subject_ID"]

# === Step 2: Define row colors by risk group ===
group_colors = {
    0: "green",
    1: "blue",
    2: "orange",
    3: "red"
}
row_colors = df["risk_for_ad"].map(group_colors)
row_colors.index = df["Subject_ID"]  # <- Sync index

# === Step 3: Clustered heatmap with subject IDs and color bar ===
g = sns.clustermap(
    embedding_matrix,
    cmap="coolwarm",
    row_cluster=True,
    col_cluster=True,
    yticklabels=True,         # Show subject IDs
    figsize=(14, 10),
    row_colors=row_colors,    # Keep risk group color bar
    cbar_kws={"label": "Embedding Value"}
)


# === Step 4: Create and place legend above the heatmap ===
legend_elements = [
    Patch(facecolor='green', label='No risk'),
    Patch(facecolor='blue', label='Familiar'),
    Patch(facecolor='orange', label='MCI'),
    Patch(facecolor='red', label='AD')
]

g.fig.legend(
    handles=legend_elements,
    title='Risk Group',
    loc='upper right',
    bbox_to_anchor=(1.10, 1.0),  # Move legend higher
    fontsize=12,
    title_fontsize=13
)



# === Step 5: Add plot title and show ===

plt.suptitle("SHAP Embedding Heatmap", fontsize=15, y=1.07)

import os

# Define full output path
save_path = r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\SHAP_CL\7_ SHAP_CL\7.6 Personalized risk profiles\figures\shap_embedding_heatmap.png"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the figure
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

