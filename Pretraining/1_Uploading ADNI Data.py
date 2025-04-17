
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