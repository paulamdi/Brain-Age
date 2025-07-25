# cBAG vs. Brain Region Volumes (Relative, z-scored) 
# No outliers
# FDR corrected

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import re, os

# === Load data ===
df_bag = pd.read_csv("BAG_with_all_metadata.csv")
vol_path = r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE DATA\AD_Decode_Regional_Stats\AD_Decode_studywide_stats_for_volume.txt"
df_vol_raw = pd.read_csv(vol_path, sep="\t").iloc[1:]
df_vol_raw = df_vol_raw[df_vol_raw["ROI"] != "0"].reset_index(drop=True)

# === Transpose to per-subject format ===
subject_cols = [col for col in df_vol_raw.columns if col.startswith("S")]
df_vol_transposed = df_vol_raw[subject_cols].transpose()
df_vol_transposed.columns = [f"Index_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

# === Clean subject IDs ===
cleaned_vol = {}
for subj in df_vol_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)
        cleaned_vol[subj_id] = df_vol_transposed.loc[subj]

df_vol_clean = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_clean.index.name = "subject_id"
df_vol_clean = df_vol_clean.reset_index()

# === Compute total brain volume and relative z-scored volumes ===
roi_cols = [col for col in df_vol_clean.columns if col.startswith("Index_")]
df_vol_clean["Total_Brain_Vol"] = df_vol_clean[roi_cols].sum(axis=1)

# === ROI dictionary ===
roi_dict = {
    "Left Cerebellum Cortex": 1,
    "Left Thalamus Proper": 2,
    "Left Caudate": 3,
    "Left Putamen": 4,
    "Left Pallidum": 5,
    "Left Hippocampus": 6,
    "Left Amygdala": 7,
    "Left Accumbens Area": 8,
    "Right Cerebellum Cortex": 9,
    "Right Thalamus Proper": 10,
    "Right Caudate": 11,
    "Right Putamen": 12,
    "Right Pallidum": 13,
    "Right Hippocampus": 14,
    "Right Amygdala": 15,
    "Right Accumbens Area": 16,
    "Left Banks of STS": 17,
    "Left Caudal Anterior Cingulate": 18,
    "Left Caudal Middle Frontal": 19,
    "Left Cuneus": 20,
    "Left Entorhinal": 21,
    "Left Fusiform": 22,
    "Left Inferior Parietal": 23,
    "Left Inferior Temporal": 24,
    "Left Isthmus Cingulate": 25,
    "Left Lateral Occipital": 26,
    "Left Lateral Orbitofrontal": 27,
    "Left Lingual": 28,
    "Left Medial Orbitofrontal": 29,
    "Left Middle Temporal": 30,
    "Left Parahippocampal": 31,
    "Left Paracentral": 32,
    "Left Pars Opercularis": 33,
    "Left Pars Orbitalis": 34,
    "Left Pars Triangularis": 35,
    "Left Pericalcarine": 36,
    "Left Postcentral": 37,
    "Left Posterior Cingulate": 38,
    "Left Precentral": 39,
    "Left Precuneus": 40,
    "Left Rostral Anterior Cingulate": 41,
    "Left Rostral Middle Frontal": 42,
    "Left Superior Frontal": 43,
    "Left Superior Parietal": 44,
    "Left Superior Temporal": 45,
    "Left Supramarginal": 46,
    "Left Frontal Pole": 47,
    "Left Temporal Pole": 48,
    "Left Transverse Temporal": 49,
    "Left Insula": 50,
    "Right Banks of STS": 51,
    "Right Caudal Anterior Cingulate": 52,
    "Right Caudal Middle Frontal": 53,
    "Right Cuneus": 54,
    "Right Entorhinal": 55,
    "Right Fusiform": 56,
    "Right Inferior Parietal": 57,
    "Right Inferior Temporal": 58,
    "Right Isthmus Cingulate": 59,
    "Right Lateral Occipital": 60,
    "Right Lateral Orbitofrontal": 61,
    "Right Lingual": 62,
    "Right Medial Orbitofrontal": 63,
    "Right Middle Temporal": 64,
    "Right Parahippocampal": 65,
    "Right Paracentral": 66,
    "Right Pars Opercularis": 67,
    "Right Pars Orbitalis": 68,
    "Right Pars Triangularis": 69,
    "Right Pericalcarine": 70,
    "Right Postcentral": 71,
    "Right Posterior Cingulate": 72,
    "Right Precentral": 73,
    "Right Precuneus": 74,
    "Right Rostral Anterior Cingulate": 75,
    "Right Rostral Middle Frontal": 76,
    "Right Superior Frontal": 77,
    "Right Superior Parietal": 78,
    "Right Superior Temporal": 79,
    "Right Supramarginal": 80,
    "Right Frontal Pole": 81,
    "Right Temporal Pole": 82,
    "Right Transverse Temporal": 83,
    "Right Insula": 84
}


# === Compute RelVol_z for each ROI ===
for roi_name, idx in roi_dict.items():
    vol_col = f"{roi_name}_Vol"
    rel_col = f"{roi_name}_RelVol"
    z_col = f"{roi_name}_RelVol_z"
    df_vol_clean[vol_col] = df_vol_clean[f"Index_{idx}"]
    df_vol_clean[rel_col] = df_vol_clean[vol_col] / df_vol_clean["Total_Brain_Vol"]
    df_vol_clean[z_col] = zscore(df_vol_clean[rel_col])

# === Merge with BAG dataframe ===
df_bag["subject_id"] = df_bag["Subject_ID"].astype(str).str.zfill(5)
merge_cols = ["subject_id"] + [f"{roi}_RelVol_z" for roi in roi_dict.keys()]
df_merged = df_bag.merge(df_vol_clean[merge_cols], on="subject_id", how="inner")

# === Output folder ===
output_dir_fdr = "cBAG_vs_ROIs_Plots_FDR"
os.makedirs(output_dir_fdr, exist_ok=True)

# === Helper function for plot titles ===
def beautify(name):
    name = name.replace("_RelVol_z", "")
    return name.replace("_", " ")

# === Regression function (can skip plotting) ===
def plot_regression(x_var, y_var, data, output_path=None, remove_outliers=True):
    df = data[[x_var, y_var]].dropna().copy()
    if remove_outliers:
        df = df[(df[x_var].abs() < 20) & (df[y_var].abs() < 3)]
    if df.empty:
        return None

    X = sm.add_constant(df[x_var])
    y = df[y_var]
    model = sm.OLS(y, X).fit()

    if output_path:
        plt.figure(figsize=(6, 5))
        sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'s': 35, 'alpha': 0.7})
        plt.title(f"{beautify(y_var)} ~ {beautify(x_var)}")
        plt.xlabel(beautify(x_var))
        plt.ylabel(beautify(y_var))
        plt.text(
            0.05, 0.95,
            f"p = {model.pvalues[1]:.3g}\nR² = {model.rsquared:.2f}\nβ = {model.params[1]:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    return {
        "X": x_var,
        "Y": y_var,
        "N": len(df),
        "R²": model.rsquared,
        "p": model.pvalues[1],
        "coef": model.params[1]
    }

# === Run regressions only for cBAG (no plotting yet) ===
results = []
for roi_name in roi_dict.keys():
    x_var = "cBAG"
    y_var = f"{roi_name}_RelVol_z"
    res = plot_regression(x_var, y_var, df_merged, output_path=None)
    if res:
        results.append(res)

# === FDR correction ===
df_results = pd.DataFrame(results)
reject, pvals_corrected, _, _ = multipletests(df_results["p"], method="fdr_bh")
df_results["p_uncorrected"] = df_results["p"]
df_results.drop(columns="p", inplace=True)
df_results["p_FDR_corrected"] = pvals_corrected
df_results["FDR_significant"] = reject

# === Save results CSV ===
df_results.to_csv(os.path.join(output_dir_fdr, "regression_cBAG_vs_ROIs_FDR_corrected.csv"), index=False)

# === Plot only significant associations ===
for _, row in df_results[df_results["FDR_significant"]].iterrows():
    x_var = row["X"]
    y_var = row["Y"]
    roi_name = y_var.replace("_RelVol_z", "")
    filename = f"{roi_name.replace(' ', '_')}_vs_{x_var}_FDRsig.png"
    save_path = os.path.join(output_dir_fdr, filename)
    plot_regression(x_var, y_var, df_merged, output_path=save_path)
