# ADDECODE COUNTING SUBJECTS

    
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

# === Set seed for reproducibility ===
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)



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

print()



############################## METADATA ##############################


print("ADDECODE METADATA\n")

# === Load metadata CSV ===
metadata_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx"
df_metadata = pd.read_excel(metadata_path)

# === Generate standardized subject IDs → 'DWI_fixed' (e.g., 123 → '00123')
df_metadata["MRI_Exam_fixed"] = (
    df_metadata["MRI_Exam"]
    .fillna(0)                           # Handle NaNs first
    .astype(int)
    .astype(str)
    .str.zfill(5)
)

# === Drop fully empty rows and those with missing DWI ===
df_metadata_cleaned = df_metadata.dropna(how='all')                       # Remove fully empty rows
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["MRI_Exam"])         # Remove rows without DWI

# === Display result ===
print(f"Metadata loaded: {df_metadata.shape[0]} rows")
print(f"After cleaning: {df_metadata_cleaned.shape[0]} rows")
print()





#################### MATCH CONNECTOMES & METADATA ####################

print(" MATCHING CONNECTOMES WITH METADATA")

# === Filter metadata to only subjects with connectomes available ===
matched_metadata = df_metadata_cleaned[
    df_metadata_cleaned["MRI_Exam_fixed"].isin(cleaned_connectomes.keys())
].copy()

print(f"Matched subjects (metadata & connectome): {len(matched_metadata)} out of {len(cleaned_connectomes)}\n")

# === Build dictionary of matched connectomes ===
matched_connectomes = {
    row["MRI_Exam_fixed"]: cleaned_connectomes[row["MRI_Exam_fixed"]]
    for _, row in matched_metadata.iterrows()
}


# === Store matched metadata as a DataFrame for further processing ===
df_matched_connectomes = matched_metadata.copy()






# df_matched_connectomes:
# → Cleaned metadata that has a valid connectome
# → Includes AD/MCI

# matched_connectomes:
# → Dictionary of connectomes that have valid metadata
# → Key: subject ID
# → Value: connectome matrix
# → Includes AD/MCI








########### PCA GENES ##########

print("PCA GENES")

import pandas as pd

# Read 
df_pca = pd.read_csv("/home/bas/Desktop/MyData/AD_DECODE/PCA_genes/PCA_human_blood_top30.csv")
print(df_pca.head())

print(df_matched_connectomes.head())



# Fix id formats

# === Fix ID format in PCA DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE_1' → 'ADDECODE1'
df_pca["ID_fixed"] = df_pca["ID"].str.upper().str.replace("_", "", regex=False)



# === Fix Subject format in metadata DataFrame ===
# Convert to uppercase and remove underscores → 'AD_DECODE1' → 'ADDECODE1'
df_matched_connectomes["IDRNA_fixed"] = df_matched_connectomes["IDRNA"].str.upper().str.replace("_", "", regex=False)




###### MATCH PCA GENES WITH METADATA############

print("MATCH PCA GENES WITH METADATA")

df_metadata_PCA_withConnectome = df_matched_connectomes.merge(df_pca, how="inner", left_on="IDRNA_fixed", right_on="ID_fixed")


#Numbers

# === Show how many healthy subjects with PCA and connectome you have
print(f" subjects with metadata connectome: {df_matched_connectomes.shape[0]}")
print()

print(f"  subjects with metadata PCA & connectome: {df_metadata_PCA_withConnectome.shape[0]}")
print()


# Get the full set of subject IDs (DWI_fixed) in healthy set
all_ids = set(df_matched_connectomes["MRI_Exam_fixed"])

# Get the subject IDs (DWI_fixed) that matched with PCA
with_pca_ids = set(df_metadata_PCA_withConnectome["MRI_Exam_fixed"])

# Compute the difference: healthy subjects without PCA
without_pca_ids = all_ids - with_pca_ids

# Filter the original healthy metadata for those subjects
df_without_pca = df_matched_connectomes[
    df_matched_connectomes["MRI_Exam_fixed"].isin(without_pca_ids)
]


# Print result
print(f" subjects with connectome but NO PCA: {df_without_pca.shape[0]}")
print()

print(df_without_pca[["MRI_Exam_fixed", "IDRNA", "IDRNA_fixed"]])


# ------------------------------------------------------------------
#  AD-DECODE  (subjects with connectome + PCA)
# ------------------------------------------------------------------
df = df_metadata_PCA_withConnectome.copy()

# Harmonise Risk label
df["Risk"] = (df["Risk"]
                .fillna("NoRisk")
                .replace(r"^\s*$", "NoRisk", regex=True))

# Map sex codes (F/M already OK)
df["Sex_Label"] = df["sex"].map({"F": "Female (F)", "M": "Male (M)"})

# === Age stats ===
age_mean = df["age"].mean()
age_std  = df["age"].std()
age_min, age_max = df["age"].min(), df["age"].max()

# === Counts & %
risk_counts = df["Risk"].value_counts()
risk_perc   = (risk_counts/len(df)*100).round(1)

sex_counts  = df["Sex_Label"].value_counts()
sex_perc    = (sex_counts/len(df)*100).round(1)

apoe_counts = df["genotype"].value_counts()
apoe_perc   = (apoe_counts/len(df)*100).round(1)

# === Print -------------------------------------------
print(f"\n=== AGE ===")
print(f"Mean ± SD : {age_mean:.2f} ± {age_std:.2f}")
print(f"Range     : [{age_min:.1f}, {age_max:.1f}]")

print("\n=== DIAGNOSTIC GROUP ===")
for grp, n in risk_counts.items():
    print(f"{grp:<8}: {n:3d} ({risk_perc[grp]}%)")

print("\n=== SEX ===")
for sx, n in sex_counts.items():
    print(f"{sx:<11}: {n:3d} ({sex_perc[sx]}%)")

print("\n=== APOE GENOTYPE ===")
for gt, n in apoe_counts.items():
    print(f"{gt:<7}: {n:3d} ({apoe_perc[gt]}%)")

# Optional: build a DataFrame summary
rows = [
    ["Age", "Mean ± SD", f"{age_mean:.2f} ± {age_std:.2f}"],
    ["Age", "Range",     f"[{age_min:.1f}, {age_max:.1f}]"],
]
rows += [["Diagnostic group", g, f"{risk_counts[g]} ({risk_perc[g]}%)"] for g in risk_counts.index]
rows += [["Sex", s, f"{sex_counts[s]} ({sex_perc[s]}%)"]           for s in sex_counts.index]
rows += [["APOE genotype", a, f"{apoe_counts[a]} ({apoe_perc[a]}%)"] for a in apoe_counts.index]

df_summary_pca = pd.DataFrame(rows, columns=["Category", "Value", "Count (%)"])
print("\n--- SUMMARY TABLE (connectome + PCA) ---")
print(df_summary_pca)
