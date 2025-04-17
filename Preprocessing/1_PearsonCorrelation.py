# Pearson correlation ADDECODE - ADNI



#################  IMPORT NECESSARY LIBRARIES  ################


import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions

import torch
import random
import numpy as np

import networkx as nx  # For graph-level metrics




# ADDECODE Data

####################### CONNECTOMES ###############################
print("ADDECODE CONNECTOMES\n")

# === Define paths ===
zip_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_connectome_act.zip"
directory_inside_zip = "connectome_act/"
connectomes = {}

# === Load connectome matrices from ZIP ===
with zipfile.ZipFile(zip_path, 'r') as z:
    for file in z.namelist():
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:
                df = pd.read_csv(f, header=None)
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")
                connectomes[subject_id] = df

print(f"Total connectome matrices loaded: {len(connectomes)}")

# === Filter out connectomes with white matter on their file name ===
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}
print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# === Extract subject IDs from filenames ===
cleaned_connectomes = {}
for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)
    if match:
        num_id = match.group(1).zfill(5)  # Ensure 5-digit IDs
        cleaned_connectomes[num_id] = v

print("Example of extracted connectome numbers:")
for key in list(cleaned_connectomes.keys())[:3]:
    print(key)
print()

############################## METADATA ##############################


print("Addecode metadata\n")

# === Load metadata CSV ===
metadata_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data_defaced.csv"
df_metadata = pd.read_csv(metadata_path)

# === Generate standardized subject IDs → 'DWI_fixed' (e.g., 123 → '00123')
df_metadata["DWI_fixed"] = (
    df_metadata["DWI"]
    .fillna(0)                           # Handle NaNs first
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# === Drop fully empty rows and those with missing DWI ===
df_metadata_cleaned = df_metadata.dropna(how='all')                       # Remove fully empty rows
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["DWI"])         # Remove rows without DWI

# === Display result ===
print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()
print("Example of 'DWI_fixed' column:")
print(df_metadata_cleaned[["DWI", "DWI_fixed"]].head())
print()



#################### MATCH CONNECTOMES & METADATA ####################

print("### MATCHING CONNECTOMES WITH METADATA")

# === Filter metadata to only subjects with connectomes available ===
matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["DWI_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

# === Build dictionary of matched connectomes ===
matched_connectomes = {
    row["DWI_fixed"]: cleaned_connectomes[row["DWI_fixed"]]
    for _, row in matched_metadata.iterrows()
}


# === Store matched metadata as a DataFrame for further processing ===
df_matched_connectomes = matched_metadata.copy()


#################### SHOW EXAMPLE CONNECTOME WITH AGE ####################

# === Display one matched connectome and its metadata ===
example_id = df_matched_connectomes["DWI_fixed"].iloc[0]
example_age = df_matched_connectomes["age"].iloc[0]
example_matrix = matched_connectomes[example_id]

print(f"Example subject ID: {example_id}")
print(f"Age: {example_age}")
print("Connectome matrix (first 5 rows):")
print(example_matrix.head())
print()

# === Plot heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(example_matrix, cmap="viridis")
plt.title(f"Connectome Heatmap - Subject {example_id} (Age {example_age})")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

#Remove AD and MCI

# === Print risk distribution if available ===
if "Risk" in df_matched_connectomes.columns:
    risk_filled = df_matched_connectomes["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()



print("FILTERING OUT AD AND MCI SUBJECTS")

# === Keep only healthy control subjects ===
df_matched_addecode_healthy = df_matched_connectomes[
    ~df_matched_connectomes["Risk"].isin(["AD", "MCI"])
].copy()

print(f"Subjects before filtering: {len(df_matched_connectomes)}")
print(f"Subjects after removing AD/MCI: {len(df_matched_addecode_healthy)}")

# === Show updated 'Risk' distribution ===
if "Risk" in df_matched_addecode_healthy.columns:
    risk_filled = df_matched_addecode_healthy["Risk"].fillna("NoRisk").replace(r'^\s*$', "NoRisk", regex=True)
    print("Risk distribution in matched data:")
    print(risk_filled.value_counts())
else:
    print("No 'Risk' column found.")
print()







# ADNI Data

###################### Connectomes ############################
print("ADNI CONNECTOMES")


import os
import pandas as pd

# Define the base path where all subject visit folders are stored
base_path_adni = "/home/bas/Desktop/Paula Pretraining/data/"

# Dictionary to store connectomes for each subject and timepoint
adni_connectomes = {}

# Loop through every folder in the base directory
for folder_name in os.listdir(base_path_adni):
    folder_path = os.path.join(base_path_adni, folder_name)

    # Only process if the current item is a directory
    if os.path.isdir(folder_path):

        # Check if the folder name ends with '_connectomics'
        if "_connectomics" in folder_name:
            # Remove the suffix to get the subject ID and timepoint (e.g., R0072_y0)
            connectome_id = folder_name.replace("_connectomics", "")

            # The expected filename inside the folder (e.g., R0072_y0_onn_plain.csv)
            file_name = f"{connectome_id}_onn_plain.csv"
            file_path = os.path.join(folder_path, file_name)

            # Check if the expected file exists
            if os.path.isfile(file_path):
                try:
                    # Load the CSV as a DataFrame without headers
                    df = pd.read_csv(file_path, header=None)

                    # Store the matrix using ID as the key (e.g., "R0072_y0")
                    adni_connectomes[connectome_id] = df

                except Exception as e:
                    # Handle any error during file loading
                    print(f"Error loading {file_path}: {e}")

# Summary: how many connectomes were successfully loaded
print("Total ADNI connectomes loaded:", len(adni_connectomes))

# Show a few example keys from the dictionary
print("Example keys:", list(adni_connectomes.keys())[:5])
print()



###################### Metadata #############################


print ("ADNI Metadata\n")

import pandas as pd

# Load metadata from Excel file
metadata_path_adni = "/home/bas/Desktop/Paula Pretraining/metadata/idaSearch_3_19_2025FINAL.xlsx"
df_adni_metadata = pd.read_excel(metadata_path_adni, sheet_name="METADATA")

# Show basic info
print("ADNI metadata loaded. Shape:", df_adni_metadata.shape)
print()




# ADD COLUMN WITH  CORRESPONDING CONNECTOME KEYS

# Extract the numeric part of the Subject ID (e.g., from "003_S_4288" → "4288")
df_adni_metadata["Subject_Num"] = df_adni_metadata["Subject ID"].str.extract(r"(\d{4})$")

# Define mapping from visit description to simplified code
visit_map = {
    "ADNI3 Initial Visit-Cont Pt": "y0",
    "ADNI3 Year 4 Visit": "y4"
}

# Map the Visit column to y0 / y4 codes
df_adni_metadata["Visit_Clean"] = df_adni_metadata["Visit"].map(visit_map)

# Remove rows with unknown or unneeded visit types
df_adni_metadata = df_adni_metadata[df_adni_metadata["Visit_Clean"].notnull()]

# Build the final connectome key for each row (e.g., "R4288_y0")
df_adni_metadata["connectome_key"] = "R" + df_adni_metadata["Subject_Num"] + "_" + df_adni_metadata["Visit_Clean"]




# KEEP ONLY ONE LINE FOR EACH SUBJECT (TWO TIMEPOINTS EACH)

# Drop duplicate connectome_key rows to keep only one per connectome
df_adni_metadata_unique = df_adni_metadata.drop_duplicates(subset="connectome_key").copy()

# Summary
print("\nTotal unique connectome keys:", df_adni_metadata_unique.shape[0])
print(df_adni_metadata_unique[["Subject ID", "Visit", "connectome_key"]].head())
print()


############# Match connectomes and metadata ##################

print("MATCHING CONNECTOMES WITH METADATA\n")

# Keep only metadata rows where the connectome key exists in the connectome dictionary
df_matched_adni = df_adni_metadata_unique[
    df_adni_metadata_unique["connectome_key"].isin(adni_connectomes.keys())
].copy()

# Show result
print("Matched connectomes:", df_matched_adni.shape[0])
print()


#Printing a connectome wit its age and subject id
 
import seaborn as sns
import matplotlib.pyplot as plt

# Select a row from the matched metadata (you can change the index)
row = df_matched_adni.sample(1).iloc[0]


# Extract subject info
subject_id = row["Subject ID"]
connectome_key = row["connectome_key"]
age = row["Age"]

# Get the connectome matrix
matrix = adni_connectomes[connectome_key]

# Print subject info and connectome
print(f"Subject ID: {subject_id}")
print(f"Connectome Key: {connectome_key}")
print(f"Age: {age}")
print("Connectome matrix (first 5 rows):")
print(matrix.head())
print()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="viridis", square=True)
plt.title(f"Connectome Heatmap - {connectome_key} (Age: {age})")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.show()


# Keeping healthy subjects

# Keep only healthy control subjects (CN)
df_matched_adni_healthy = df_matched_adni[df_matched_adni["Research Group"] == "CN"].copy()

# Show result
print("Number of healthy ADNI subjects with matched connectomes:", df_matched_adni_healthy.shape[0])
print()
print()




# PEARSON CORRELATION

# === GROUP BY AGE RANGE AND COMPUTE AVERAGE CONNECTOMES ===

#ADDECODE

print("ADDECODE subjects grouped by age\n")

# Manually define fixed age bins
age_bins = [
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90)
]

# Dictionary to store average connectome vectors for each age range
mean_vectors_by_bin_addecode = {}

for low, high in age_bins:
    print(f"\nProcessing age group {low}-{high}")

    # Filter subjects within this age range
    df_bin = df_matched_connectomes[
        (df_matched_connectomes["age"] >= low) &
        (df_matched_connectomes["age"] < high)
    ].copy()

    # List of subject IDs in this bin
    bin_ids = df_bin["DWI_fixed"].tolist()

    # Extract upper triangle vector from each connectome
    flattened_vectors = []
    for subj_id in bin_ids:
        if subj_id in matched_connectomes:
            matrix = matched_connectomes[subj_id].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            flattened_vectors.append(upper_tri)

    if len(flattened_vectors) == 0:
        print(f"  No valid subjects found in this group.")
        continue

    # Stack and compute the average vector for the group
    group_array = np.stack(flattened_vectors)
    mean_vector = np.mean(group_array, axis=0)

    label = f"{low}-{high}"
    mean_vectors_by_bin_addecode[label] = mean_vector

    print(f"  Mean connectome vector stored for group {label}. Shape: {mean_vector.shape}")

print()
print()









#ADni

print("ADNI subjects grouped by age\n")


import numpy as np



# Dictionary to store mean vectors for each age group
mean_vectors_by_bin_adni = {}

for low, high in age_bins:
    print(f"\nProcessing ADNI age group {low}-{high}")

    # Filter ADNI healthy subjects within the age range
    df_bin = df_matched_adni_healthy[
        (df_matched_adni_healthy["Age"] >= low) &
        (df_matched_adni_healthy["Age"] < high)
    ].copy()

    # Get list of connectome keys for this age bin
    bin_keys = df_bin["connectome_key"].tolist()

    # Extract upper triangle vectors from each connectome
    flattened_vectors = []
    for key in bin_keys:
        if key in adni_connectomes:
            matrix = adni_connectomes[key].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            flattened_vectors.append(upper_tri)

    if len(flattened_vectors) == 0:
        print(f"  No valid subjects found in this group.")
        continue

    # Stack and average the vectors
    group_array = np.stack(flattened_vectors)
    mean_vector = np.mean(group_array, axis=0)

    label = f"{low}-{high}"
    mean_vectors_by_bin_adni[label] = mean_vector

    print(f"  Mean connectome vector stored for group {label}. Shape: {mean_vector.shape}")






# Pearson correlation beetween ADDECODE for each age group

from scipy.stats import pearsonr
from itertools import combinations
import numpy as np


# Store results
within_group_correlations = {}

print("\n=== INTRA-GROUP PEARSON CORRELATION (AD-DECODE) ===")

for low, high in age_bins:
    # Filter subjects in this age range
    df_group = df_matched_connectomes[
        (df_matched_connectomes["age"] >= low) &
        (df_matched_connectomes["age"] < high)
    ].copy()

    subject_ids = df_group["DWI_fixed"].tolist()

    # Extract vectors
    subject_vectors = {}
    for subj in subject_ids:
        if subj in matched_connectomes:
            matrix = matched_connectomes[subj].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            subject_vectors[subj] = upper_tri

    # Calculate all pairwise Pearson correlations
    correlations = []
    for subj1, subj2 in combinations(subject_vectors.keys(), 2):
        r, _ = pearsonr(subject_vectors[subj1], subject_vectors[subj2])
        correlations.append(r)

    # Store and print results
    label = f"{low}-{high}"
    if correlations:
        mean_r = np.mean(correlations)
        std_r = np.std(correlations)
        within_group_correlations[label] = (mean_r, std_r, len(correlations))
        print(f"Age group {label} → Mean r = {mean_r:.4f}, Std = {std_r:.4f}, Pairs = {len(correlations)}")
    else:
        print(f"Age group {label} → Not enough subjects to compute correlations")



# Pearson correlation beetween ADNI for each age group


# Store results
within_group_correlations_adni = {}

print("\n=== INTRA-GROUP PEARSON CORRELATION (ADNI) ===")

for low, high in age_bins:
    # Filter ADNI healthy subjects in this age range
    df_group = df_matched_adni_healthy[
        (df_matched_adni_healthy["Age"] >= low) &
        (df_matched_adni_healthy["Age"] < high)
    ].copy()

    connectome_keys = df_group["connectome_key"].tolist()

    # Extract vectors
    subject_vectors = {}
    for key in connectome_keys:
        if key in adni_connectomes:
            matrix = adni_connectomes[key].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            subject_vectors[key] = upper_tri

    # Calculate all pairwise Pearson correlations
    correlations = []
    for k1, k2 in combinations(subject_vectors.keys(), 2):
        r, _ = pearsonr(subject_vectors[k1], subject_vectors[k2])
        correlations.append(r)

    # Store and print results
    label = f"{low}-{high}"
    if correlations:
        mean_r = np.mean(correlations)
        std_r = np.std(correlations)
        within_group_correlations_adni[label] = (mean_r, std_r, len(correlations))
        print(f"Age group {label} → Mean r = {mean_r:.4f}, Std = {std_r:.4f}, Pairs = {len(correlations)}")
    else:
        print(f"Age group {label} → Not enough subjects to compute correlations")




# Pearson correlation ADDECODE - ADNI 


print("\n=== PEARSON CORRELATION BETWEEN AD-DECODE AND ADNI ===\n")

# Determine which age bins exist in both datasets
common_bins = set(mean_vectors_by_bin_addecode.keys()) & set(mean_vectors_by_bin_adni.keys())

for age_bin in sorted(common_bins):
    vec_addecode = mean_vectors_by_bin_addecode[age_bin]
    vec_adni = mean_vectors_by_bin_adni[age_bin]

    # Compute Pearson correlation
    r, p = pearsonr(vec_addecode, vec_adni)

    print(f"Age group {age_bin} → Pearson r = {r:.4f}, p = {p:.4e}")




#Subject vs subject

from scipy.stats import pearsonr
from itertools import product
import numpy as np


# Store results
cross_group_correlations = {}

print("\n=== CROSS-DATASET PEARSON CORRELATION (AD-DECODE vs ADNI) ===")

for low, high in age_bins:
    # Filter AD-DECODE subjects in this age range
    df_addecode = df_matched_connectomes[
        (df_matched_connectomes["age"] >= low) &
        (df_matched_connectomes["age"] < high)
    ].copy()

    ids_addecode = df_addecode["DWI_fixed"].tolist()

    # Extract AD-DECODE vectors
    subject_vectors_dec = {}
    for subj in ids_addecode:
        if subj in matched_connectomes:
            matrix = matched_connectomes[subj].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            subject_vectors_dec[subj] = upper_tri

    # Filter ADNI subjects in this age range
    df_adni = df_matched_adni_healthy[
        (df_matched_adni_healthy["Age"] >= low) &
        (df_matched_adni_healthy["Age"] < high)
    ].copy()

    ids_adni = df_adni["connectome_key"].tolist()

    # Extract ADNI vectors
    subject_vectors_adni = {}
    for key in ids_adni:
        if key in adni_connectomes:
            matrix = adni_connectomes[key].to_numpy()
            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
            subject_vectors_adni[key] = upper_tri

    # Calculate all cross-pair Pearson correlations
    correlations = []
    for subj_dec, subj_adni in product(subject_vectors_dec.keys(), subject_vectors_adni.keys()):
        r, _ = pearsonr(subject_vectors_dec[subj_dec], subject_vectors_adni[subj_adni])
        correlations.append(r)

    # Store and print results
    label = f"{low}-{high}"
    if correlations:
        mean_r = np.mean(correlations)
        std_r = np.std(correlations)
        cross_group_correlations[label] = (mean_r, std_r, len(correlations))
        print(f"Age group {label} → Mean r = {mean_r:.4f}, Std = {std_r:.4f}, Pairs = {len(correlations)}")
    else:
        print(f"Age group {label} → Not enough subjects to compute correlations")
        
        
        
