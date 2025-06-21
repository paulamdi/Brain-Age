#PLOT SHAP EDGES dti


# 1) PERSONALIZED 
# TOP 10 connections for a young middle and old person, 


import pandas as pd
import matplotlib.pyplot as plt
import os

# === Region names list ===
region_names = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

# === Function to plot and save SHAP edges ===
def load_and_plot_top_edges(subject_id, subject_age, shap_folder="shap_outputs", save=True):
    # === Load SHAP values
    path_csv = os.path.join(shap_folder, f"edge_shap_dti_subject_{subject_id}.csv")
    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()
    df["abs_shap"] = df["SHAP_value"].abs()
    top_edges = df.sort_values(by="abs_shap", ascending=False).head(15)

    # === Build edge labels
    edge_labels = [
        f"{region_names[i]} ↔ {region_names[j]}"
        for i, j in zip(top_edges["Node_i"], top_edges["Node_j"])
    ]
    shap_vals = top_edges["SHAP_value"]

    # === Plot
    plt.figure(figsize=(9, 5))
    plt.barh(edge_labels, shap_vals, color=shap_vals.apply(lambda x: "steelblue" if x > 0 else "crimson"))
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("SHAP value (edge contribution to predicted age)")
    plt.title(f"Top SHAP Edge Importances — {subject_id} (Age: {int(subject_age)})")
    plt.tight_layout()
    plt.gca().invert_yaxis()

    # === Save or show
    if save:
        save_dir = os.path.join("SHAP edges", "personalised")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"SHAP_edges_{subject_id}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved plot: {save_path}")
    else:
        plt.show()

# === Generate plots for selected subjects
load_and_plot_top_edges("ADRC0069", subject_age=28)
load_and_plot_top_edges("ADRC0095", subject_age=56)
load_and_plot_top_edges("ADRC0044", subject_age=77)




# 2) TOP 10 beeswarm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Region names (84) ===
region_names = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]

# === Input/output paths ===
shap_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/shap_outputs"
output_dir = "/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/5.2 shap edges dti/SHAP edges/beeswarm"
os.makedirs(output_dir, exist_ok=True)

# === Load all DTI SHAP CSVs ===
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.startswith("edge_shap_dti_subject_") and fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        df["subject"] = fname.replace("edge_shap_dti_subject_", "").replace(".csv", "")
        shap_dfs.append(df)

if not shap_dfs:
    raise ValueError("No SHAP CSVs found in directory. Check path or filenames.")

df_all = pd.concat(shap_dfs, ignore_index=True)
df_all["abs_SHAP_value"] = df_all["SHAP_value"].abs()

# === Top 10 most important edges
mean_shap = df_all.groupby(["Node_i", "Node_j"])["abs_SHAP_value"].mean().reset_index()
top10_edges = mean_shap.sort_values(by="abs_SHAP_value", ascending=False).head(10)
top_pairs = set((row.Node_i, row.Node_j) for _, row in top10_edges.iterrows())

# === Filter only top edges and name them
df_top = df_all[df_all.apply(lambda row: (row.Node_i, row.Node_j) in top_pairs, axis=1)].copy()
df_top["Edge"] = df_top.apply(lambda row: f"{region_names[row.Node_i]} ↔ {region_names[row.Node_j]}", axis=1)

# === Plot beeswarm
plt.figure(figsize=(10, 6))
sns.stripplot(data=df_top, x="SHAP_value", y="Edge", jitter=True, alpha=0.6, size=4)
plt.title("Top 10 Most Important DTI Edges (SHAP)")
plt.xlabel("SHAP Value")
plt.ylabel("Edge")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

# === Save
save_path = os.path.join(output_dir, "top10_dti_edges_beeswarm.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(" Saved beeswarm to:", save_path)




