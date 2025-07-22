# BAG vs. Hippocampal Volume (Relative, z-scored)
# ===============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
from scipy.stats import zscore

# === Load BAG predictions (includes BAG, cBAG, age, metadata) ===
df_bag = pd.read_csv("/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/BAG_with_all_metadata.csv")

# === Load raw regional volume data ===
vol_path = "/home/bas/Desktop/MyData/AD_DECODE/RegionalStats/AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume.txt"
df_vol_raw = pd.read_csv(vol_path, sep="\t").iloc[1:]  # Remove header row used as data
df_vol_raw = df_vol_raw[df_vol_raw["ROI"] != "0"].reset_index(drop=True)

# === Extract subject columns and transpose for per-subject format ===
subject_cols = [col for col in df_vol_raw.columns if col.startswith("S")]
df_vol_transposed = df_vol_raw[subject_cols].transpose()
df_vol_transposed.columns = [f"Index_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

# === Clean and standardize subject IDs to 5-digit format ===
cleaned_vol = {}
for subj in df_vol_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)  # Extract e.g. "02842"
        cleaned_vol[subj_id] = df_vol_transposed.loc[subj]

# === Reconstruct volume DataFrame with subject ID index ===
df_vol_clean = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_clean.index.name = "subject_id"

# === Extract raw hippocampal volumes (based on ROI indices 6 and 14) ===
df_vol_clean["Left_Hippocampus_Vol"] = df_vol_clean["Index_6"]
df_vol_clean["Right_Hippocampus_Vol"] = df_vol_clean["Index_14"]
df_vol_clean = df_vol_clean.reset_index()  # Make subject_id a column

# === Calculate total brain volume by summing all 84 ROIs per subject ===
roi_cols = [col for col in df_vol_clean.columns if col.startswith("Index_")]
df_vol_clean["Total_Brain_Vol"] = df_vol_clean[roi_cols].sum(axis=1)

# === Compute relative hippocampal volumes ===
df_vol_clean["Left_Hippocampus_RelVol"] = df_vol_clean["Left_Hippocampus_Vol"] / df_vol_clean["Total_Brain_Vol"]
df_vol_clean["Right_Hippocampus_RelVol"] = df_vol_clean["Right_Hippocampus_Vol"] / df_vol_clean["Total_Brain_Vol"]

# === Z-score the relative volumes for comparability ===
df_vol_clean["Left_Hippocampus_RelVol_z"] = zscore(df_vol_clean["Left_Hippocampus_RelVol"])
df_vol_clean["Right_Hippocampus_RelVol_z"] = zscore(df_vol_clean["Right_Hippocampus_RelVol"])

# === Format subject ID in BAG dataframe to 5-digit string ===
df_bag_copy = df_bag.copy()
df_bag_copy["subject_id"] = df_bag_copy["Subject_ID"].astype(str).str.zfill(5)

# === Merge relative hippocampal volumes into BAG dataframe ===
df_bag_with_vol = df_bag_copy.merge(
    df_vol_clean[
        ["subject_id",
         "Left_Hippocampus_Vol", "Right_Hippocampus_Vol",
         "Left_Hippocampus_RelVol_z", "Right_Hippocampus_RelVol_z"]
    ],
    on="subject_id",
    how="inner"
)

# === Preview merged result ===
print("Merged shape:", df_bag_with_vol.shape)
print(df_bag_with_vol[[
    "Subject_ID", "Left_Hippocampus_Vol", "Right_Hippocampus_Vol",
    "Left_Hippocampus_RelVol_z", "Right_Hippocampus_RelVol_z"
]].head())

# === Define regression plotting function ===
def plot_regression(x_var, y_var, data):
    """
    Run linear regression of y ~ x and plot results.
    Includes R² and p-value annotation.
    """
    df = data[[x_var, y_var]].dropna()
    if df.empty:
        print(f"No data for {x_var} vs. {y_var}")
        return None

    X = sm.add_constant(df[x_var])
    y = df[y_var]
    model = sm.OLS(y, X).fit()

    # Plot regression
    plt.figure(figsize=(5.5, 4))
    sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'s': 35, 'alpha': 0.7})
    plt.title(f"{y_var} ~ {x_var}")
    plt.xlabel(x_var)
    plt.ylabel("Hippocampal Volume (z-scored rel.)")
    plt.text(
        0.05, 0.95,
        f"p = {model.pvalues[1]:.3g}, R² = {model.rsquared:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )
    plt.tight_layout()
    plt.show()

    return {
        "X": x_var,
        "Y": y_var,
        "N": len(df),
        "R²": model.rsquared,
        "p": model.pvalues[1],
        "coef": model.params[1]
    }

# === Run BAG/cBAG vs. relative hippocampal volume regressions ===
results = []
for bag_type in ["BAG", "cBAG"]:
    for side in ["Left", "Right"]:
        # Use z-scored relative volume as Y
        y_var = f"{side}_Hippocampus_RelVol_z"
        res = plot_regression(bag_type, y_var, df_bag_with_vol)
        if res:
            results.append(res)

# === Save regression summary to CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv("regression_BAG_vs_Rel_z_hippocampus.csv", index=False)
print("Saved results to regression_BAG_vs_Rel_z_hippocampus.csv")
