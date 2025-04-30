import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import nibabel as nib
from nilearn import plotting

# ==== LOAD REGION COORDINATES FROM ATLAS ====
nii_path = "/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_labels.nii.gz"
lookup_path = "/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_lookup.xlsx"

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

# ==== CONFIG ====
csv_files = [
    ("/home/bas/Desktop/Paula Pretraining/Codes/Transfer Learning/BEST (copy)/subject_age_20_layer_gnn1.csv", "Age 20"),
    ("/home/bas/Desktop/Paula Pretraining/Codes/Transfer Learning/BEST (copy)/subject_age_45_layer_gnn1.csv", "Age 45"),
    ("/home/bas/Desktop/Paula Pretraining/Codes/Transfer Learning/BEST (copy)/subject_age_78_layer_gnn1.csv", "Age 78")
]

# ==== PLOT PER SUBJECT ====
for file, label in csv_files:
    df = pd.read_csv(file)

    # Keep top 10
    df_top = df.sort_values("attention_score", ascending=False).head(10)

    # Regions involved
    regions = sorted(set(df_top["region_source"]) | set(df_top["region_target"]))
    region_to_index = {name: i for i, name in enumerate(regions)}
    coords = [region_name_to_coords[r] for r in regions]

    # Build matrix
    n = len(regions)
    matrix = np.zeros((n, n))
    for _, row in df_top.iterrows():
        i = region_to_index[row["region_source"]]
        j = region_to_index[row["region_target"]]
        matrix[i, j] = row["attention_score"]
        matrix[j, i] = row["attention_score"]

    # === PLOT BRAIN WITH LEGEND ===
    cmap = cm.get_cmap('tab10', n)
    node_colors = [cmap(i) for i in range(n)]

    fig = plt.figure(figsize=(14, 6))
    ax_brain = fig.add_axes([0.05, 0.05, 0.7, 0.9])
    ax_legend = fig.add_axes([0.77, 0.2, 0.2, 0.6])

    display = plotting.plot_connectome(
        matrix,
        coords,
        edge_threshold="0%",
        node_color=node_colors,
        node_size=100,
        edge_cmap=plt.cm.Reds,
        display_mode='ortho',
        axes=ax_brain,
        title=None
    )

    fig.suptitle(f"Top 10 Attention Connections – {label} – GNN Layer 1", fontsize=24, color='black')

    ax_legend.axis('off')
    legend_patches = [
        mpatches.Patch(color=node_colors[i], label=f"{i:02d} → {region}")
        for i, region in enumerate(regions)
    ]
    ax_legend.legend(handles=legend_patches, loc='center left', fontsize=14, frameon=False)

    plt.tight_layout()
    plt.show()


