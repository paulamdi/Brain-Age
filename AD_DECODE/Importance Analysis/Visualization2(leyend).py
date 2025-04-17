# VISUALIZE top10 CONECTIONS -16 nodes , LOGGED, TH

#################  LOAD TOP CONNECTIONS CSV  ################

import pandas as pd

# Load the CSV file containing edge importance per patient
csv_path = "/home/bas/Desktop/Paula/Importance 16-/top_connections_by_patient_MASKED.csv"
df = pd.read_csv(csv_path)

#################  GROUP BY CONNECTION AND AVERAGE DELTA  ################

import matplotlib.pyplot as plt

# Group by unique pairs of brain regions and calculate the mean Δ across patients
grouped = df.groupby(["Region_1", "Region_2"])["Delta"].mean().reset_index()

# Create a string representation of each connection for plotting
grouped["Connection"] = grouped["Region_1"] + " ↔ " + grouped["Region_2"]

# Sort by mean Δ and select the top 10 most influential connections
top_mean_df = grouped.sort_values(by="Delta", ascending=False).head(10)

# Plot a horizontal bar chart of the top 10 connections
plt.figure(figsize=(10, 6))
plt.barh(top_mean_df["Connection"], top_mean_df["Delta"], color='orange')
plt.gca().invert_yaxis()  # Most important at the top
plt.title("Top 10 Most Influential Connections (Mean Δ)")
plt.xlabel("Average Δ Prediction")
plt.tight_layout()
plt.show()

#################  EXTRACT CENTROIDS AND MERGE WITH NAMES  ################

import nibabel as nib
import numpy as np

# Load the label image (.nii.gz) from the IIT atlas
img = nib.load("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_labels.nii.gz")
data = img.get_fdata()
affine = img.affine

# Get all unique non-zero region labels
region_labels = np.unique(data)
region_labels = region_labels[region_labels != 0]

# Calculate the centroid (mean voxel coordinate) for each region
centroids = []
for label in region_labels:
    mask = data == label                # select all voxels that belong to the current region
    coords = np.argwhere(mask)         # get the voxel coordinates [i, j, k] of that region
    center_voxel = coords.mean(axis=0) # compute the average voxel position = centroid (in voxel space)
    center_mni = nib.affines.apply_affine(affine, center_voxel)  # convert centroid to real-world MNI space

    centroids.append(center_mni)

# Store centroid coordinates and region label in a DataFrame
centroid_df = pd.DataFrame(centroids, columns=["X", "Y", "Z"])
centroid_df["Label"] = region_labels.astype(int)

# Load region names and label mappings from Excel lookup table
lookup = pd.read_excel("/home/bas/Desktop/Paula/Visualization/IITmean_RPI/IITmean_RPI_lookup.xlsx")

# Merge centroid coordinates with anatomical region names
final_df = pd.merge(centroid_df, lookup, left_on="Label", right_on="Index")

# Keep only relevant columns
final_df = final_df[["Structure", "Label", "X", "Y", "Z"]]

# Print a preview of the resulting table
print(final_df.head())

#################  BUILD REGION NAME TO COORDINATE MAP  ################

# Create a dictionary mapping region names to MNI coordinates
region_name_to_coords = {
    row["Structure"]: [row["X"], row["Y"], row["Z"]] for _, row in final_df.iterrows()
}

#################  VISUALIZE TOP 10 CONNECTIONS ON GLASS BRAIN  ################

from nilearn import plotting

# Get all unique regions involved in the top 10 connections
regions_involved = list(set(top_mean_df["Region_1"]) | set(top_mean_df["Region_2"]))

#################  PRINT COORDINATES OF REGIONS IN TOP 10 CONNECTIONS  ################

print("\nCoordinates of regions involved in top 10 most influential connections:\n")

# Print MNI coordinates of each region involved in the top 10
for region in sorted(regions_involved):
    coords_xyz = region_name_to_coords.get(region)
    if coords_xyz:
        print(f"{region:35} →  X: {coords_xyz[0]:.2f}, Y: {coords_xyz[1]:.2f}, Z: {coords_xyz[2]:.2f}")
    else:
        print(f"{region:35} →    Not found in coordinate map")

# Map each region to an index for building the connectivity matrix
region_to_index = {region: idx for idx, region in enumerate(regions_involved)}

# Build a list of coordinates (ordered by region index)
coords = [region_name_to_coords[region] for region in regions_involved]

# Create an NxN empty symmetric matrix for edge weights (Δ values)
n = len(regions_involved)
con_matrix = np.zeros((n, n))

# Fill the connectivity matrix with mean Δ values from the top 10 edges
for _, row in top_mean_df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    con_matrix[i, j] = row["Delta"]
    con_matrix[j, i] = row["Delta"]  # symmetric

# Plot the connectome over glass brain (without thresholding)
display = plotting.plot_connectome(con_matrix, coords, edge_threshold=None)

plotting.show()





#################  PRINT COORDINATES OF EACH TOP CONNECTION  ################

print("\nTop 10 connections with MNI coordinates of each region:\n")

# Print detailed info for each of the top 10 connections with coordinates
for i, row in top_mean_df.iterrows():
    r1 = row["Region_1"]
    r2 = row["Region_2"]
    delta = row["Delta"]
    c1 = region_name_to_coords.get(r1, None)
    c2 = region_name_to_coords.get(r2, None)

    print(f"{i+1:02d}. {r1} ↔ {r2}")
    print(f"    → {r1.split('-')[-1]} at X: {c1[0]:.1f}, Y: {c1[1]:.1f}, Z: {c1[2]:.1f}")
    print(f"    → {r2.split('-')[-1]} at X: {c2[0]:.1f}, Y: {c2[1]:.1f}, Z: {c2[2]:.1f}")
    print(f"    Δ = {delta:.4f}\n")
    
    
print()
    


#################  GLASS BRAIN WITH TITLE ABOVE  ################

from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# Set up node colors using a colormap
num_nodes = len(regions_involved)
cmap = cm.get_cmap('tab10', num_nodes)
node_colors = [cmap(i) for i in range(num_nodes)]

# Plot the connectome WITHOUT title
display = plotting.plot_connectome(
    con_matrix,
    coords,
    edge_threshold="0%",  # Show all edges
    node_color=node_colors,
    node_size=80,
    edge_cmap=plt.cm.Reds,
    title=None  # No title here
)

# Add clean external title above (in black on white)
fig = plt.gcf()
fig.suptitle("Top 10 Brain Connections", fontsize=14, color='black', y=0.95)

# Show the figure
plotting.show()



#################  DISPLAY NODE LEGEND AS TEXT ################

print("\n Node index to region mapping:\n")
for idx, region in enumerate(regions_involved):
    print(f"{idx:02d} → {region}")


#################  VISUAL COLOR LEGEND FOR NODE INDEX  ################

import matplotlib.patches as mpatches

# Build list of colored patches
legend_patches = []
for idx, region in enumerate(regions_involved):
    patch = mpatches.Patch(color=node_colors[idx], label=f"{idx:02d} → {region}")
    legend_patches.append(patch)

# Plot the color legend as standalone figure
plt.figure(figsize=(5, len(legend_patches) * 0.4))
plt.legend(handles=legend_patches, loc='center left', frameon=False)
plt.axis('off')
plt.title("Node Color Legend")
plt.tight_layout()
plt.show()





#############  LEYEND IN BRAIN IMAGE #####################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nilearn import plotting
import numpy as np
import matplotlib.cm as cm

#################  PREPARE COLORS ################

num_nodes = len(regions_involved)
cmap = cm.get_cmap('tab10', num_nodes)
node_colors = [cmap(i) for i in range(num_nodes)]

#################  CREATE BIGGER CANVAS WITH CUSTOM AXES ################

fig = plt.figure(figsize=(14, 6))  # más ancho

# Custom axes positions (left = brain, right = legend)
ax_brain = fig.add_axes([0.05, 0.05, 0.7, 0.9])  # [left, bottom, width, height]
ax_legend = fig.add_axes([0.77, 0.2, 0.2, 0.6])  # más pequeña y centrada

# Glass brain plot on custom axes
display = plotting.plot_connectome(
    con_matrix,
    coords,
    edge_threshold="0%",
    node_color=node_colors,
    node_size=100,
    edge_cmap=plt.cm.Reds,
    display_mode='ortho',
    axes=ax_brain,
    title=None
)

# Title
fig.suptitle("Top 10 Brain Connections", fontsize=30, color='black')

# Legend
ax_legend.axis('off')
legend_patches = [
    mpatches.Patch(color=node_colors[i], label=f"{i:02d} → {region}")
    for i, region in enumerate(regions_involved)
]
ax_legend.legend(
    handles=legend_patches,
    loc='center left',
    fontsize=18,
    frameon=False
)

plt.show()
