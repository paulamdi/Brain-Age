import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import zscore
import os

# === Load BAG results (with Subject_ID as key) ===
df_bag = pd.read_csv("/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/brainage_predictions_adrc_all_clipped120.csv")   

# === Load volume data (already preprocessed like in your node features pipeline) ===
vol_path = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/ADRC_Regional_Stats/studywide_stats_for_volume.txt"
df_vol = pd.read_csv(vol_path, sep="\t")
df_vol = df_vol[df_vol["ROI"] != 0].reset_index(drop=True)

# === Load valid subjects only ===
valid_subjects = set(df_bag["Subject_ID"].astype(str))
subject_cols = [col for col in df_vol.columns if col in valid_subjects]

# === Transpose to get [subjects × 84 ROIs]
df_vol_transposed = df_vol[subject_cols].transpose()
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

# === Compute total brain volume
roi_cols = df_vol_transposed.columns
df_vol_transposed["Total_Brain_Vol"] = df_vol_transposed[roi_cols].sum(axis=1)

# === Define ROIs of interest (14 regions)
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

# === Compute relative and z-score volumes

for roi_name, idx in roi_dict.items():
    vol_col = f"{roi_name}_Vol"
    rel_col = f"{roi_name}_RelVol"
    z_col = f"{roi_name}_RelVol_z"

    df_vol_transposed[vol_col] = df_vol_transposed[idx - 1]  
    df_vol_transposed[rel_col] = df_vol_transposed[vol_col] / df_vol_transposed["Total_Brain_Vol"]
    df_vol_transposed[z_col] = zscore(df_vol_transposed[rel_col])

# === Reset index and prepare for merge
df_vol_transposed = df_vol_transposed.reset_index().rename(columns={"subject_id": "Subject_ID"})
df_vol_transposed["Subject_ID"] = df_vol_transposed["Subject_ID"].astype(str)

# === Merge with BAG
df_bag["Subject_ID"] = df_bag["Subject_ID"].astype(str)
df_merged = df_bag.merge(df_vol_transposed, on="Subject_ID", how="inner")

# === Output folder
out_dir = "regression_BAG_vs_ROIs_ADRC"
os.makedirs(out_dir, exist_ok=True)




# === Helper function to format axis labels and titles ===
def prettify_label(label):
    if "_RelVol_z" in label:
        roi = label.replace("_RelVol_z", "").replace("_", " ")
        return f"{roi} (Relative Volume z-score)"
    return label.replace("_", " ")




# === Regression function
def plot_regression(x_var, y_var, data, out_name):
    df = data[[x_var, y_var]].dropna()
    df = df[(df[x_var].abs() < 20) & (df[y_var].abs() < 3)]

    if df.empty:
        print(f"Skipping: {y_var} ~ {x_var}")
        return None

    X = sm.add_constant(df[x_var])
    y = df[y_var]
    model = sm.OLS(y, X).fit()

    # === Inside plot_regression() ===
    plt.figure(figsize=(5.5, 4))
    sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'s': 35, 'alpha': 0.7})
    
    # === Format axis labels ===
    roi_name = y_var.replace("_RelVol_z", "").replace("_", " ")
    plt.title(f"{roi_name} vs. {x_var}")
    x_label = x_var
    y_label = f"{roi_name} Relative Volume (z-scored)"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # === Add regression stats as text ===
    plt.text(
        0.7, 0.1,
        f"p-value = {model.pvalues[1]:.3g}\nR² = {model.rsquared:.2f}\nβ = {model.params[1]:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close()


    return {
        "X": x_var,
        "Y": y_var,
        "N": len(df),
        "R²": model.rsquared,
        "p-value": model.pvalues[1],
        "β": model.params[1]
    }

# === Run all regressions
results = []
for bag_type in ["BAG", "cBAG"]:
    for roi_name in roi_dict.keys():
        y_var = f"{roi_name}_RelVol_z"
        out_name = f"{y_var}_vs_{bag_type}.png"
        print(f"Running: {y_var} ~ {bag_type}")
        res = plot_regression(bag_type, y_var, df_merged, out_name)
        if res:
            results.append(res)

# === Save summary
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(out_dir, "regression_BAG_vs_allROIs_ADRC.csv"), index=False)
print("Saved: regression_BAG_vs_allROIs_ADRC.csv")
