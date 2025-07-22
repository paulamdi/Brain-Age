
import pandas as pd

# === Load the CSV file with SHAP values and global metadata
df_shap = pd.read_csv(r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\global_shap_outputs\shap_global_features_all_subjects.csv")




# 1) BEESWARM FOR ALL SUBJECTS 


# 1a. ONE FOR ALL global features: graph metrics and PCA genes 
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === 1. Load SHAP values
df_shap = pd.read_csv(
    r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\global_shap_outputs\shap_global_features_all_subjects.csv"
)

# If Subject_ID is in index, restore it as column
if "Subject_ID" not in df_shap.columns and df_shap.index.name == "Subject_ID":
    df_shap.reset_index(inplace=True)

# === 2. Load real feature values
df_real = pd.read_csv(
    r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\brain_age_predictions_with_metadata.csv"
)
# Codificar 'sex' como 0 (F) y 1 (M)
df_real["sex"] = df_real["sex"].map({"F": 0, "M": 1})

# Rename ID column to match and drop duplicates
df_real = df_real.rename(columns={"MRI_Exam_fixed": "Subject_ID"})
df_real = df_real.drop_duplicates(subset="Subject_ID")

# === 3. Set index for alignment
df_shap = df_shap.set_index("Subject_ID")
df_real = df_real.set_index("Subject_ID")

# === 4. Keep only common subjects
common_ids = df_shap.index.intersection(df_real.index)
df_shap_final = df_shap.loc[common_ids]
df_real_final = df_real.loc[common_ids]

# === 5. Define feature names
feature_names = [
    "Systolic", "Diastolic", "sex_encoded", "genotype",
    "Clustering_Coeff", "Path_Length",
    "PC12", "PC7", "PC13", "PC5", "PC21", "PC14", "PC1", "PC16", "PC17", "PC3"
]

# Map SHAP name to real name if needed
column_name_map = {
    "sex_encoded": "sex"
}
real_feature_names = [column_name_map.get(name, name) for name in feature_names]

# === 6. Extract matrices
shap_matrix = df_shap_final[feature_names].values
real_feature_matrix = df_real_final[real_feature_names].values

# === 7. Check shapes
print(" SHAP matrix shape:", shap_matrix.shape)
print(" Real feature matrix shape:", real_feature_matrix.shape)

# === 8. Create SHAP Explanation
shap_values_all = shap.Explanation(
    values=shap_matrix,
    data=real_feature_matrix,
    feature_names=feature_names
)

# === 9. Create output folder if not exists
output_dir = r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\figures"
os.makedirs(output_dir, exist_ok=True)

# === 10. Plot and save beeswarm
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values_all, max_display=16, show=False)
plt.title("SHAP Beeswarm — All Subjects (Correct Color)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "beeswarm_all_subjects.png"), dpi=300)
plt.show()

print(" Beeswarm plot saved to:", os.path.join(output_dir, "beeswarm_all_subjects.png"))








# 1b. One per kind of global feature

import shap
import pandas as pd
import matplotlib.pyplot as plt

# === Load SHAP values
df_shap = pd.read_csv(r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\global_shap_outputs\shap_global_features_all_subjects.csv")

# === Define feature groups
demographic_cols = ["Systolic", "Diastolic", "sex_encoded", "genotype"]
graphmetric_cols = ["Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3']

# === Helper function to plot
def plot_beeswarm(df, features, title):
    shap_matrix = df[features].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=real_feature_matrix,
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Plot per group
plot_beeswarm(df_shap, demographic_cols, "SHAP Beeswarm — Demographic Features")
plot_beeswarm(df_shap, graphmetric_cols, "SHAP Beeswarm — Connectome Graph Metrics")
plot_beeswarm(df_shap, pca_cols, "SHAP Beeswarm — PCA Gene Components")




# CHECK COLOR OF DOTS 
# 2) BEESWARM BY AGE GROUPS

# SHAP BY AGE GROUPS

#Create thre age groups
# Create age bins using tertiles
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# Optional: check distribution
print(df_shap["Age_Group"].value_counts())


#Graph function
import shap
import matplotlib.pyplot as plt

# === Get list of SHAP feature names (skip Subject_ID, Age, Age_Group)
feature_names = df_shap.columns.difference(["Subject_ID", "Age", "Age_Group"]).tolist()

def plot_beeswarm(df_group, title):
    shap_matrix = df_group[feature_names].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,
        feature_names=feature_names
    )
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(feature_names), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


#Use function
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    plot_beeswarm(df_group, f"SHAP Beeswarm — Age Group: {group}")





# 3) BEESWARM BY AGE GROUP AND TYPE OF FEATURE


import shap
import pandas as pd
import matplotlib.pyplot as plt

# === Load SHAP with Age and group by tertiles
df_shap = pd.read_csv(r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\global_shap_outputs\shap_global_features_all_subjects.csv")
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# === Define feature groups
demographic_cols = ["Systolic", "Diastolic", "sex_encoded", "genotype"]
graphmetric_cols = ["Clustering_Coeff", "Path_Length"]
pca_cols = ['PC12', 'PC7', 'PC13', 'PC5', 'PC21', 'PC14', 'PC1', 'PC16', 'PC17', 'PC3']

# === Beeswarm plotting function
def plot_beeswarm(df, features, title):
    shap_matrix = df[features].values
    shap_exp = shap.Explanation(
        values=shap_matrix,
        data=shap_matrix,  # Optional: to keep scale
        feature_names=features
    )
    plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_exp, max_display=len(features), show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === Loop over age groups and plot each group separately
for group in ["Young", "Middle", "Old"]:
    df_group = df_shap[df_shap["Age_Group"] == group]
    
    # Plot per feature group
    plot_beeswarm(df_group, demographic_cols, f"SHAP Beeswarm — Demographics ({group})")
    plot_beeswarm(df_group, graphmetric_cols, f"SHAP Beeswarm — Graph Metrics ({group})")
    plot_beeswarm(df_group, pca_cols, f"SHAP Beeswarm — PCA Genes ({group})")





# 4) PERSONALIZED global feature Importance per Subject
# One for a young person, one for a middle aged person and one for an older person



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load SHAP values for all subjects (with Age column) ===
df_shap = pd.read_csv(r"C:\Users\Paula\OneDrive\Escritorio\CODES DUKE\ADDECODE_PCAs\SHAP ADDECODE\global_shap_outputs\shap_global_features_all_subjects.csv")

# === Drop unnecessary columns to get feature names ===
feature_names = df_shap.columns.difference(["Subject_ID", "Age"]).tolist()

# === Create age tertiles: Young, Middle, Old ===
df_shap["Age_Group"] = pd.qcut(df_shap["Age"], q=3, labels=["Young", "Middle", "Old"])

# === Pick one subject from each age group ===

# === Select youngest, middle, and oldest subject in df_shap ===
subject_young = df_shap.loc[df_shap["Age"].idxmin()]   # Youngest subject
subject_old = df_shap.loc[df_shap["Age"].idxmax()]     # Oldest subject

# For middle, select the one closest to the median
median_age = df_shap["Age"].median()
subject_middle = df_shap.iloc[(df_shap["Age"] - median_age).abs().argsort().iloc[0]]


# === Helper function to plot top N SHAP global features for one subject ===
import matplotlib.pyplot as plt

# === New plotting function ===
def plot_subject_shap_signed(subject_row):
    subject_id = subject_row["Subject_ID"]
    age = subject_row["Age"]
    
    # Extract SHAP values for all 16 features (with sign)
    shap_values = subject_row[feature_names]

    # Sort by absolute value, but keep sign
    shap_sorted = shap_values.reindex(shap_values.abs().sort_values(ascending=True).index)

    # === Plot horizontal bar chart (signed values) ===
    plt.figure(figsize=(7, 5))
    shap_sorted.plot(kind="barh", color=shap_sorted.apply(lambda x: "crimson" if x < 0 else "steelblue"))
    plt.axvline(0, color="black", linestyle="--", linewidth=0.7)
    plt.xlabel("SHAP value (contribution to prediction)")
    plt.title(f"SHAP Global Feature Impact — Subject {subject_id} (Age: {int(age)})")
    plt.tight_layout()
    plt.show()


# === Apply to one subject per group ===
plot_subject_shap_signed(subject_young)
plot_subject_shap_signed(subject_middle)
plot_subject_shap_signed(subject_old)


