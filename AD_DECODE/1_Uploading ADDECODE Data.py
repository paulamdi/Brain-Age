# ADDECODE Uploading Connectomes, Metadata,  matching and keeping healthy subjects

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

# === Print original risk distribution if available ===
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


# === Filter connectomes to include only those from non-AD/MCI subjects ===
matched_connectomes_healthy_addecode = {
    row["DWI_fixed"]: matched_connectomes[row["DWI_fixed"]]
    for _, row in df_matched_addecode_healthy.iterrows()
}

# === Confirmation of subject count
print(f"Connectomes selected (excluding AD/MCI): {len(matched_connectomes_healthy_addecode)}")



# df_matched_connectomes:
# → Cleaned metadata that has a valid connectome
# → Includes AD/MCI

# matched_connectomes:
# → Dictionary of connectomes that have valid metadata
# → Key: subject ID
# → Value: connectome matrix
# → Includes AD/MCI
