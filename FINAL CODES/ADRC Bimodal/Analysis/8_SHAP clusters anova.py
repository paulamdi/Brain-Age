# Loads SHAP global feature values and brain age predictions (BAG, cBAG) for each subject.
# Applies KMeans clustering to group subjects based on their SHAP feature profiles.
# Merges cluster labels with brain age prediction data.
# Runs ANOVA and Kruskal-Wallis tests to compare BAG and cBAG across SHAP-based clusters.
# Generates and saves boxplots and statistical results to visualize cluster-wise differences.





import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import kruskal

# === Paths ===
shap_path = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/shap_global_features_adrc_bimodal.csv"  
prediction_path = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/brainage_predictions_adrc_all_clipped120.csv"
output_dir = "shap_cluster_analysis"
os.makedirs(output_dir, exist_ok=True)

# === Load SHAP global features ===
shap_df = pd.read_csv(shap_path)

# === Drop Subject_ID to keep only features ===
subject_ids = shap_df["Subject_ID"]
X = shap_df.drop(columns=["Subject_ID"])

# === KMeans Clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
shap_df["SHAP_Cluster"] = kmeans.fit_predict(X)

# === Merge with BAG/cBAG predictions ===
bag_df = pd.read_csv(prediction_path)
merged_df = shap_df.merge(bag_df, left_on="Subject_ID", right_on="Subject_ID")

# === Save merged data with cluster labels ===
merged_df.to_csv(os.path.join(output_dir, "merged_shap_clusters.csv"), index=False)

# === Run ANOVA and Kruskal-Wallis on BAG and cBAG ===
for metric in ["BAG", "cBAG"]:
    print(f"\n=== {metric} by SHAP Cluster ===")
    
    # Drop NaNs
    df_sub = merged_df[["SHAP_Cluster", metric]].dropna()

    # ANOVA
    model = ols(f"{metric} ~ C(SHAP_Cluster)", data=df_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    
    print(anova_table)
    anova_table.to_csv(os.path.join(output_dir, f"anova_{metric}_by_cluster.csv"))

    # Kruskal-Wallis
    groups = [df_sub[df_sub["SHAP_Cluster"] == c][metric] for c in sorted(df_sub["SHAP_Cluster"].unique())]
    stat, p_kw = kruskal(*groups)
    print(f"Kruskal-Wallis: H = {stat:.3f}, p = {p_kw:.4f}")

    # === Boxplot ===
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="SHAP_Cluster", y=metric, data=df_sub, palette="Set2")
    plt.xlabel("SHAP-based Cluster")
    plt.ylabel(metric)
    plt.title(f"{metric} by SHAP Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_boxplot_by_cluster.png"))
    plt.show()

