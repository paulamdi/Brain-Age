import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import nibabel as nib
from nilearn import plotting

# === LOAD REGION COORDINATES FROM ATLAS ===
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





# === LOAD ATTENTION DATA FROM ONE SUBJECT (AGE 45, GNN4) ===
csv_path = "/home/bas/Desktop/Paula Pretraining/Codes/Transfer Learning/BEST (copy)/subject_age_45_layer_gnn4.csv"  # ← Path to the CSV file with attention scores and region pairs
df = pd.read_csv(csv_path)  # ← Load the CSV into a pandas DataFrame

# === HELPER FUNCTIONS ===
def side_of_region(region):  # ← Helper to determine if a brain region is left or right hemisphere
    if region.startswith("Left") or region.startswith("ctx-lh-"):
        return "left"  # ← Left subcortical or cortical region
    elif region.startswith("Right") or region.startswith("ctx-rh-"):
        return "right"  # ← Right subcortical or cortical region
    else:
        return "unknown"  # ← Catch-all in case the region name doesn’t match

def categorize_connection(row):  # ← Classify the connection type based on source and target hemispheres
    src = side_of_region(row["region_source"])  # ← Get hemisphere for source node
    tgt = side_of_region(row["region_target"])  # ← Get hemisphere for target node
    if src == tgt == "left":
        return "left"  # ← Both nodes in the left hemisphere → intra-left
    elif src == tgt == "right":
        return "right"  # ← Both nodes in the right hemisphere → intra-right
    elif src != tgt and "unknown" not in (src, tgt):
        return "inter"  # ← Nodes on different hemispheres → interhemispheric
    return "unknown"  # ← Fallback in case of unknown region

# === CLASSIFY CONNECTIONS ===
df["category"] = df.apply(categorize_connection, axis=1)  # ← Apply categorization function row-by-row and create a new column "category"

# === SELECT TOP 5 PER CATEGORY ===
top5_left = df[df["category"] == "left"].sort_values("attention_score", ascending=False).head(5)  # ← Select top-5 strongest intra-left connections
top5_right = df[df["category"] == "right"].sort_values("attention_score", ascending=False).head(5)  # ← Select top-5 strongest intra-right connections
top5_inter = df[df["category"] == "inter"].sort_values("attention_score", ascending=False).head(5)  # ← Select top-5 strongest interhemispheric connections

# === COMBINE ALL ===
df_top = pd.concat([top5_left, top5_right, top5_inter]).reset_index(drop=True)  # ← Combine all 15 into a single DataFrame and reset the index



# Build the matrix and region coordinates

# === GET UNIQUE REGIONS INVOLVED IN df_top ===
regions = sorted(set(df_top["region_source"]) | set(df_top["region_target"]))  # ← Get all unique regions from source and target
region_to_index = {name: i for i, name in enumerate(regions)}  # ← Map each region to a matrix index (0 to N-1)

# === GET COORDINATES FOR THE REGIONS ===
coords = [region_name_to_coords[r] for r in regions]  # ← Get MNI coordinates for each region

# === BUILD ATTENTION MATRIX ===
n = len(regions)
matrix = np.zeros((n, n))  # ← Initialize square matrix (n x n)

for _, row in df_top.iterrows():
    i = region_to_index[row["region_source"]]
    j = region_to_index[row["region_target"]]
    score = row["attention_score"]
    matrix[i, j] = score
    matrix[j, i] = score  # ← Make it symmetric for visualization


# Plot the glass brain




