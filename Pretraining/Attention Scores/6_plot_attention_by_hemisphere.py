import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import nibabel as nib
from nilearn import plotting
import os

# ==== LOAD REGION COORDINATES FROM ATLAS ====
nii_path = "/home/bas/Desktop/MyData/AD_DECODE/IITmean_RPI/IITmean_RPI_labels.nii.gz"
lookup_path = "/home/bas/Desktop/MyData/AD_DECODE/IITmean_RPI/IITmean_RPI_lookup.xlsx"

img = nib.load(nii_path)
data = img.get_fdata()
affine = img.affine

region_labels = np.unique(data)
region_labels = region_labels[region_labels != 0]

centroids = []
for label in region_labels:
    mask = data == label
    coords = np.argwhere(mask)
    center_voxel = coords.mean(axis=0)
    center_mni = nib.affines.apply_affine(affine, center_voxel)
    centroids.append(center_mni)

centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
centroid_df["Label"] = region_labels.astype(int)

lookup_df = pd.read_excel(lookup_path)
final_df = pd.merge(centroid_df, lookup_df, left_on="Label", right_on="Index")
region_name_to_coords = {row["Structure"]: [row["X"], row["Y"], row["Z"]] for _, row in final_df.iterrows()}

# ==== DEFINE PARAMETERS ====
ages = [20, 45, 78]
layers = ["gnn1", "gnn4"]
base_csv_path = "/home/bas/Desktop/Paula Pretraining/Codes/Transfer Learning/BEST (copy)/"

# ==== FUNCTIONS ====
def side_of_region(region):
    if region.startswith("Left") or region.startswith("ctx-lh-"):
        return "left"
    elif region.startswith("Right") or region.startswith("ctx-rh-"):
        return "right"
    return "unknown"

def categorize_connection(row):
    src = side_of_region(row["region_source"])
    tgt = side_of_region(row["region_target"])
    if src == tgt == "left":
        return "left"
    elif src == tgt == "right":
        return "right"
    elif src != tgt and "unknown" not in (src, tgt):
        return "inter"
    return "unknown"

# ==== LOOP THROUGH SUBJECTS AND LAYERS ====
for age in ages:
    for layer in layers:
        csv_path = os.path.join(base_csv_path, f"subject_age_{age}_layer_{layer}.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["category"] = df.apply(categorize_connection, axis=1)

        top10_left = df[df["category"] == "left"].sort_values("attention_score", ascending=False).head(10)
        top10_right = df[df["category"] == "right"].sort_values("attention_score", ascending=False).head(10)
        top10_inter = df[df["category"] == "inter"].sort_values("attention_score", ascending=False).head(10)

        fig = plt.figure(figsize=(14, 6))
        ax_brain = fig.add_axes([0.05, 0.05, 0.7, 0.9])
        ax_legend = fig.add_axes([0.78, 0.2, 0.2, 0.6])

        all_regions = sorted(set(top10_left["region_source"]) | set(top10_left["region_target"]) |
                             set(top10_right["region_source"]) | set(top10_right["region_target"]) |
                             set(top10_inter["region_source"]) | set(top10_inter["region_target"]))

        region_to_index = {name: i for i, name in enumerate(all_regions)}
        try:
            coords = [region_name_to_coords[r] for r in all_regions]
        except KeyError as e:
            print(f"Missing region in atlas: {e}")
            continue

        n = len(all_regions)
        node_colors = [plt.colormaps['tab20'](i % 20) for i in range(n)]

        display = plotting.plot_connectome(
            np.zeros((n, n)),
            coords,
            node_color=node_colors,
            node_size=100,
            display_mode='ortho',
            axes=ax_brain,
            edge_threshold=None,
            title=None
        )

        category_dfs = {"left": top10_left, "right": top10_right, "inter": top10_inter}
        category_colors = {"left": "blue", "right": "red", "inter": "green"}

        for category, group_df in category_dfs.items():
            adj_matrix = np.zeros((n, n))
            for _, row in group_df.iterrows():
                i = region_to_index[row["region_source"]]
                j = region_to_index[row["region_target"]]
                adj_matrix[i, j] = row["attention_score"]
                adj_matrix[j, i] = row["attention_score"]

            display.add_graph(adj_matrix, coords, edge_kwargs={"color": category_colors[category], "linewidth": 2})

        ax_legend.axis('off')
        ax_legend.text(0.0, 1.07, "Connection types", fontsize=12, fontweight='bold', va='top')
        for i, (label, color) in enumerate(zip(["Left", "Right", "Inter"], ["blue", "red", "green"])):
            y = 0.98 - i * 0.07
            ax_legend.add_patch(plt.Rectangle((0.0, y - 0.02), 0.03, 0.03, color=color, transform=ax_legend.transAxes))
            ax_legend.text(0.05, y, label + " connections", fontsize=11, va='center', transform=ax_legend.transAxes)

        ax_legend.text(0.0, 0.78, "Node colors\n", fontsize=12, fontweight='bold', va='top')
        for i, (region, color) in enumerate(zip(all_regions, node_colors)):
            row = i % 15
            col = i // 15
            y = 0.68 - row * 0.055
            x = 0.0 + col * 0.55
            ax_legend.add_patch(plt.Circle((x, y), radius=0.02, color=color, transform=ax_legend.transAxes))
            ax_legend.text(x + 0.03, y, f"{i:02d} → {region}", fontsize=11, va='center', transform=ax_legend.transAxes)

        fig.suptitle(f"Top 10 Connections – Age {age} – Layer {layer.upper()}", fontsize=18)
        plt.tight_layout()
        plt.show()

        # Print connections to terminal
        print(f"\n=== SUBJECT AGE {age} – LAYER {layer.upper()} ===")
        for category, df_group in category_dfs.items():
            print(f"→ {category.upper()} CONNECTIONS (Total: {len(df_group)}):")
            for _, row in df_group.iterrows():
                print(f"   {row['region_source']} → {row['region_target']} | Attention: {row['attention_score']:.4f}")
            print()
