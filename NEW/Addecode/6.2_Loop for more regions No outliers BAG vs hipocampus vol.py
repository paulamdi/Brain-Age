# BAG vs. Brain Region Volumes (Relative, z-scored) - Extended Version with Multiple ROIs
# =============================================================

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

df_vol_clean = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_clean.index.name = "subject_id"
df_vol_clean = df_vol_clean.reset_index()  # Make subject_id a column

# === Calculate total brain volume by summing all 84 ROIs per subject ===
roi_cols = [col for col in df_vol_clean.columns if col.startswith("Index_")]
df_vol_clean["Total_Brain_Vol"] = df_vol_clean[roi_cols].sum(axis=1)

# === Define brain regions to analyze ===
roi_dict = {
    "Left_Cerebellum_Cortex": 1,
    "Left_Thalamus": 2,
    "Left_Caudate": 3,
    "Left_Putamen": 4,
    "Left_Pallidum": 5,
    "Left_Hippocampus": 6,
    "Left_Amygdala": 7,
    "Right_Cerebellum_Cortex": 9,
    "Right_Thalamus": 10,
    "Right_Caudate": 11,
    "Right_Putamen": 12,
    "Right_Pallidum": 13,
    "Right_Hippocampus": 14,
    "Right_Amygdala": 15
}


# === Compute relative volumes and z-scores for each ROI ===
for roi_name, idx in roi_dict.items():
    vol_col = f"{roi_name}_Vol"
    rel_col = f"{roi_name}_RelVol"
    z_col = f"{roi_name}_RelVol_z"

    df_vol_clean[vol_col] = df_vol_clean[f"Index_{idx}"]
    df_vol_clean[rel_col] = df_vol_clean[vol_col] / df_vol_clean["Total_Brain_Vol"]
    df_vol_clean[z_col] = zscore(df_vol_clean[rel_col])

# === Format subject ID in BAG dataframe to 5-digit string ===
df_bag_copy = df_bag.copy()
df_bag_copy["subject_id"] = df_bag_copy["Subject_ID"].astype(str).str.zfill(5)

# === Merge 
cols_to_merge = ["subject_id"]

for roi in roi_dict.keys():
    cols_to_merge.append(f"{roi}_Vol")
    cols_to_merge.append(f"{roi}_RelVol_z")

df_bag_with_vol = df_bag_copy.merge(df_vol_clean[cols_to_merge], on="subject_id", how="inner")


# === Define regression plotting function ===
def plot_regression(x_var, y_var, data, remove_outliers=True):
    df = data[[x_var, y_var]].dropna().copy()

    if remove_outliers:
        before = len(df)
        df = df[(df[x_var].abs() < 20) & (df[y_var].abs() < 3)]
        after = len(df)
        print(f"Removed {before - after} outliers")

    if df.empty:
        print(f"No valid data for regression {y_var} ~ {x_var}")
        return None

    X = sm.add_constant(df[x_var])
    y = df[y_var]
    model = sm.OLS(y, X).fit()

    plt.figure(figsize=(5.5, 4))
    sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'s': 35, 'alpha': 0.7})
    plt.title(f"{y_var} ~ {x_var}")
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.text(
        0.05, 0.95,
        f"p = {model.pvalues[1]:.3g}, R² = {model.rsquared:.2f}\nβ = {model.params[1]:.3f}",
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

# === Run regression for all combinations of BAG type and brain region ===
results = []
for bag_type in ["BAG", "cBAG"]:
    for roi_name in roi_dict.keys():
        x_var = bag_type
        y_var = f"{roi_name}_RelVol_z"
        print(f"Running regression: {y_var} ~ {x_var}")
        res = plot_regression(x_var, y_var, df_bag_with_vol)
        if res:
            results.append(res)

# === Save results ===
df_results = pd.DataFrame(results)
df_results.to_csv("regression_BAG_vs_allROIs.csv", index=False)
print("Saved results to regression_BAG_vs_allROIs.csv")