


# === Personal SHAP Edge Plot (AD-DECODE) ===

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

# === Load and plot function ===
def plot_top_edges_addecode(subject_id, subject_age, shap_folder=r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\shap_outputs_addecode", save=True):
    path_csv = os.path.join(shap_folder, f"edge_shap_subject_{subject_id}.csv")
    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()
    df["abs_shap"] = df["SHAP_val"].abs()
    top_edges = df.sort_values("abs_shap", ascending=False).head(15)

    # Build edge labels
    labels = [
        f"{region_names[i]} ↔ {region_names[j]}"
        for i, j in zip(top_edges["Node_i"], top_edges["Node_j"])
    ]
    shap_vals = top_edges["SHAP_val"]

    # Plot
    plt.figure(figsize=(9, 5))
    plt.barh(labels, shap_vals,
             color=shap_vals.apply(lambda x: "steelblue" if x > 0 else "crimson"))
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("SHAP value (edge contribution to predicted age)")
    plt.title(f"Top SHAP Edges — {subject_id} (Age: {int(subject_age)})")
    plt.tight_layout()
    plt.gca().invert_yaxis()

    # Save
    if save:
        out_dir = os.path.join("SHAP edges1", "personalised")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"SHAP_edges_{subject_id}.png"), dpi=300)
        plt.close()
        print(f"✔ Saved plot for {subject_id}")
    else:
        plt.show()

# === Example: pick one young, one middle, one older subject ===
plot_top_edges_addecode("02231", subject_age=21)
plot_top_edges_addecode("02473", subject_age=52)
plot_top_edges_addecode("02967", subject_age=82)



# === Beeswarm plot of top-10 most important edges (AD-DECODE) ===

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Paths ===
shap_dir = r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\shap_outputs_addecode"
output_dir = "SHAP edges1/beeswarm"
os.makedirs(output_dir, exist_ok=True)

# === Load all SHAP CSVs ===
shap_dfs = []
for fname in os.listdir(shap_dir):
    if fname.endswith(".csv") and fname.startswith("edge_shap_subject_"):
        df = pd.read_csv(os.path.join(shap_dir, fname))
        df["subject"] = fname.replace("edge_shap_subject_", "").replace(".csv", "")
        shap_dfs.append(df)

# === Concatenate and compute mean absolute SHAP ===
df_all = pd.concat(shap_dfs, ignore_index=True)
df_all["abs_SHAP"] = df_all["SHAP_val"].abs()

# === Identify top-10 edges by mean SHAP ===
mean_shap = df_all.groupby(["Node_i", "Node_j"])["abs_SHAP"].mean().reset_index()
top10 = mean_shap.sort_values("abs_SHAP", ascending=False).head(10)
top_pairs = set(zip(top10["Node_i"], top10["Node_j"]))

# === Filter and label ===
df_top = df_all[df_all.apply(lambda row: (row["Node_i"], row["Node_j"]) in top_pairs, axis=1)].copy()
df_top["Edge"] = df_top.apply(lambda row: f"{region_names[row.Node_i]} ↔ {region_names[row.Node_j]}", axis=1)

# === Plot beeswarm ===
plt.figure(figsize=(10, 6))
sns.stripplot(data=df_top, x="SHAP_val", y="Edge", jitter=True, alpha=0.6, size=4)
plt.title("Top 10 Most Important DTI Edges (AD-DECODE)")
plt.xlabel("SHAP Value")
plt.ylabel("Edge")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

# === Save
plt.savefig(os.path.join(output_dir, "top10_dti_edges_beeswarm_addecode.png"), dpi=300)
plt.close()
print("✔ Saved beeswarm plot.")

