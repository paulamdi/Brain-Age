#With standarscaler
# === umap_shap_embeddings.py ===
#Visualize how different groups cluster
"""
UMAP Visualization of SHAP-based Embeddings

Steps:
1. Load SHAP-based embeddings (from contrastive learning)
2. Load subject-level metadata (e.g., age, risk group, APOE, etc.)
3. Merge embeddings and metadata using Subject_ID
4. Apply UMAP to project high-dimensional embeddings into 2D
5. Visualize with seaborn scatterplots, colored by various variables
"""

# === Import required libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import umap  # UMAP = Uniform Manifold Approximation and Projection

# === 1. Load SHAP embeddings ===
# These embeddings were generated using your contrastive learning model on SHAP values
df_embed = pd.read_csv("shap_embeddings.csv")  # Should contain 'Subject_ID' and dim_0, dim_1, ...

# === 2. Load subject metadata (used for coloring plots) ===
df_meta = pd.read_excel("/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx")  # Contains age, diagnosis, APOE, etc.

# === 3. Merge embeddings and metadata ===
# We merge using Subject_ID to align embeddings with clinical info
df = df_embed.merge(df_meta, left_on="Subject_ID", right_on="MRI_Exam")


# === 4. Extract embedding matrix ===
# Select only the embedding dimensions (e.g., dim_0, dim_1, ..., dim_31)
embed_cols = [col for col in df.columns if col.startswith("embed_")]

X = df[embed_cols].values  # Numpy matrix of shape (n_subjects, embed_dim)


from sklearn.preprocessing import StandardScaler
# Scale
X_scaled = StandardScaler().fit_transform(X)




# === 5. Apply UMAP for dimensionality reduction ===
# This will reduce the embeddings to 2D for visualization
reducer = umap.UMAP(random_state=42)  # Fix seed for reproducibility
X_umap = reducer.fit_transform(X_scaled)  # Output shape: (n_subjects, 2)

# Add UMAP coordinates back to the DataFrame
df["UMAP1"] = X_umap[:, 0]
df["UMAP2"] = X_umap[:, 1]

# === 6. Define function to plot UMAP and color by metadata column ===
def plot_umap_by(column, palette="Set2", save=False):
    """
    Visualize UMAP embeddings colored by a given metadata column.

    Args:
        column (str): column name to color the points by (e.g., 'Age', 'RiskGroup')
        palette (str): seaborn color palette
        save (bool): if True, save the figure as PNG
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="UMAP1", y="UMAP2",
        hue=column, palette=palette,
        s=60, alpha=0.9, edgecolor="k", linewidth=0.3
    )
    plt.title(f"UMAP of SHAP Embeddings colored by: {column}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save:
        filename = f"umap_by_{column}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")
    
    plt.show()

# === 7. Example usage: plot by different metadata variables ===
plot_umap_by("age", palette="viridis", save=True)           # Continuous

plot_umap_by("APOE", palette="coolwarm", save=True)  # 0, 1, 2 copies of e4
plot_umap_by("genotype", palette="coolwarm", save=True)
plot_umap_by("sex", palette="Set1", save=True)              # Male/Female

# === UMAP colored by risk group with custom legend labels ===

# Map numerical risk levels to descriptive labels
risk_labels = {
    0: "No risk",
    1: "Familial",
    2: "MCI",
    3: "AD"
}

# Replace values in a new column for display
df["Risk_Label"] = df["risk_for_ad"].map(risk_labels)

# Plot with seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="UMAP1", y="UMAP2",
    hue="Risk_Label", palette="Set2",
    s=60, alpha=0.9, edgecolor="k", linewidth=0.3
)
plt.title("UMAP of SHAP Embeddings colored by Risk Group")
plt.legend(title="Risk Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
