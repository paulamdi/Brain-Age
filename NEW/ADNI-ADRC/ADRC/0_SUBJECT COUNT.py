#ADRC DTI fmri 2 channels bimodal 
#with rmse
#With biomarkers


import os
import pandas as pd
import numpy as np
import random
import torch



import os

output_dir = "bimodal_training_eval_plots"
os.makedirs(output_dir, exist_ok=True)




def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


#LOAD CONNECTOMES
print ("LOAD CONNECTOMES DTI")

base_path_adrc_dti = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/DTI/plain"
adrc_dti_connectomes = {}

for filename in os.listdir(base_path_adrc_dti):
    # Skip hidden system files or ._macOS files
    if not filename.endswith("_conn_plain.csv") or filename.startswith("._"):
        continue

    subject_id = filename.replace("_conn_plain.csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_dti, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_dti_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC DTI connectomes loaded: {len(adrc_dti_connectomes)}")
print(" Example subject IDs:", list(adrc_dti_connectomes.keys())[:5])
print()





print("LOAD CONNECTOMES fMRI")

base_path_adrc_fmri = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/connectome/fMRI/corr"
adrc_fmri_connectomes = {}

for filename in os.listdir(base_path_adrc_fmri):
    if not filename.startswith("func_connectome_corr_ADRC") or not filename.endswith(".csv"):
        continue

    subject_id = filename.replace("func_connectome_corr_", "").replace(".csv", "")  # e.g., ADRC0001
    file_path = os.path.join(base_path_adrc_fmri, filename)
    try:
        matrix = pd.read_csv(file_path, header=None)
        adrc_fmri_connectomes[subject_id] = matrix
    except Exception as e:
        print(f"Error loading {filename}: {e}")

print(f" Total ADRC fMRI connectomes loaded: {len(adrc_fmri_connectomes)}")
print(" Example subject IDs:", list(adrc_fmri_connectomes.keys())[:5])
print()



# Intersect subjects that have both DTI and fMRI
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

print(f" Matched subjects with both DTI and fMRI: {len(matched_dti_fmri)}")
print(" Example matched subjects:", matched_dti_fmri[:5])
print()



#LOAD METADATA
print("LOAD METADATA ")


# === Step 1: Define the metadata file path ===
metadata_path_adrc = "/home/bas/Desktop/MyData/ADRC/data/ADRC_connectome_bank/metadata/alex-badea_2024-06-14 (copy).xlsx"

# === Step 2: Load the Excel file into a DataFrame ===
# This will read the first sheet by default
df_adrc_meta= pd.read_excel(metadata_path_adrc)

# === Step 3: Display metadata summary ===
print(" ADRC metadata loaded successfully.")
print(" Metadata shape:", df_adrc_meta.shape)
print(" Preview of first rows:")
print(df_adrc_meta.head())
print()






#MATCH CONNECTMES (has both) AND METADATA
print("=== MATCHING CONNECTOMES AND METADATA (DTI + fMRI only) ===")

# List of subjects that have both DTI and fMRI connectomes
matched_dti_fmri = sorted(set(adrc_dti_connectomes.keys()) & set(adrc_fmri_connectomes.keys()))

# Get subject IDs present in the metadata
metadata_subject_ids = set(df_adrc_meta["PTID"])

# Find subjects that have DTI, fMRI, and metadata
matched_ids = sorted(set(matched_dti_fmri) & metadata_subject_ids)

# Filter metadata to include only matched subjects
df_matched_adrc = df_adrc_meta[df_adrc_meta["PTID"].isin(matched_ids)].copy()

# Filter DTI connectomes to matched subjects
adrc_dti_connectomes_matched = {
    sid: adrc_dti_connectomes[sid]
    for sid in matched_ids
}

# Filter fMRI connectomes to matched subjects
adrc_fmri_connectomes_matched = {
    sid: adrc_fmri_connectomes[sid]
    for sid in matched_ids
}

# Print summary of matching
print(f"Subjects with DTI + fMRI: {len(matched_dti_fmri)}")
print(f"Subjects with metadata: {len(metadata_subject_ids)}")
print(f"Matched subjects (DTI + fMRI + metadata): {len(matched_ids)}")
print(f"- Rows in matched metadata: {df_matched_adrc.shape[0]}")
print(f"- DTI connectomes matched: {len(adrc_dti_connectomes_matched)}")
print(f"- fMRI connectomes matched: {len(adrc_fmri_connectomes_matched)}")
print("Example matched subject IDs:", matched_ids[:5])
print()




#Count subjects

import pandas as pd

# === Recode sex column (1 = Female, 2 = Male) ===
sex_map = {1: "Female (F)", 2: "Male (M)"}
df_matched_adrc["Sex_Label"] = df_matched_adrc["SUBJECT_SEX"].map(sex_map)  # cambia "SEXO" si tu columna tiene otro nombre

# === Age stats ===
age_mean = df_matched_adrc["SUBJECT_AGE_SCREEN"].mean()
age_std = df_matched_adrc["SUBJECT_AGE_SCREEN"].std()
age_min = df_matched_adrc["SUBJECT_AGE_SCREEN"].min()
age_max = df_matched_adrc["SUBJECT_AGE_SCREEN"].max()

# === Assign diagnostic group ===
def assign_group(row):
    if row.get("DEMENTED", 0) == 1:
        return "DEMENTED"
    elif row.get("IMPNOMCI", 0) == 1:
        return "MCI"
    elif row.get("NORMCOG", 0) == 1:
        return "NORMCOG"
    else:
        return "UNKNOWN"
df_matched_adrc["DiagGroup"] = df_matched_adrc.apply(assign_group, axis=1)

# === Counts and percentages ===
group_counts = df_matched_adrc["DiagGroup"].value_counts()
group_perc = (group_counts / len(df_matched_adrc) * 100).round(1)

sex_counts = df_matched_adrc["Sex_Label"].value_counts()
sex_perc = (sex_counts / len(df_matched_adrc) * 100).round(1)

apoe_counts = df_matched_adrc["APOE"].value_counts()
apoe_perc = (apoe_counts / len(df_matched_adrc) * 100).round(1)

# === Print results ===
print("=== AGE ===")
print(f"Mean ± SD: {age_mean:.2f} ± {age_std:.2f}")
print(f"Range: [{age_min:.1f}, {age_max:.1f}]")
print()

print("=== DIAGNOSTIC GROUP ===")
for k, v in group_counts.items():
    print(f"{k}: {v} ({group_perc[k]}%)")
print()

print("=== SEX ===")
for k, v in sex_counts.items():
    print(f"{k}: {v} ({sex_perc[k]}%)")
print()

print("=== APOE GENOTYPE ===")
for k, v in apoe_counts.items():
    print(f"{k}: {v} ({apoe_perc[k]}%)")

