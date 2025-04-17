import seaborn as sns
import matplotlib.pyplot as plt  # also needed to show the plot if you want to use plt.show()


region_names = [
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
    "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Right-Cerebellum-Cortex", "Right-Thalamus-Proper",
    "Right-Caudate", "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area",
    "ctx-lh-bankssts", "ctx-lh-caudalanteriorcingulate", "ctx-lh-caudalmiddlefrontal", "ctx-lh-cuneus",
    "ctx-lh-entorhinal", "ctx-lh-fusiform", "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate", "ctx-lh-lateraloccipital", "ctx-lh-lateralorbitofrontal", "ctx-lh-lingual",
    "ctx-lh-medialorbitofrontal", "ctx-lh-middletemporal", "ctx-lh-parahippocampal", "ctx-lh-paracentral",
    "ctx-lh-parsopercularis", "ctx-lh-parsorbitalis", "ctx-lh-parstriangularis", "ctx-lh-pericalcarine",
    "ctx-lh-postcentral", "ctx-lh-posteriorcingulate", "ctx-lh-precentral", "ctx-lh-precuneus",
    "ctx-lh-rostralanteriorcingulate", "ctx-lh-rostralmiddlefrontal", "ctx-lh-superiorfrontal",
    "ctx-lh-superiorparietal", "ctx-lh-superiortemporal", "ctx-lh-supramarginal", "ctx-lh-frontalpole",
    "ctx-lh-temporalpole", "ctx-lh-transversetemporal", "ctx-lh-insula", "ctx-rh-bankssts", "ctx-rh-caudalanteriorcingulate",
    "ctx-rh-caudalmiddlefrontal", "ctx-rh-cuneus", "ctx-rh-entorhinal", "ctx-rh-fusiform", "ctx-rh-inferiorparietal",
    "ctx-rh-inferiortemporal", "ctx-rh-isthmuscingulate", "ctx-rh-lateraloccipital", "ctx-rh-lateralorbitofrontal",
    "ctx-rh-lingual", "ctx-rh-medialorbitofrontal", "ctx-rh-middletemporal", "ctx-rh-parahippocampal",
    "ctx-rh-paracentral", "ctx-rh-parsopercularis", "ctx-rh-parsorbitalis", "ctx-rh-parstriangularis",
    "ctx-rh-pericalcarine", "ctx-rh-postcentral", "ctx-rh-posteriorcingulate", "ctx-rh-precentral", "ctx-rh-precuneus",
    "ctx-rh-rostralanteriorcingulate", "ctx-rh-rostralmiddlefrontal", "ctx-rh-superiorfrontal",
    "ctx-rh-superiorparietal", "ctx-rh-superiortemporal", "ctx-rh-supramarginal", "ctx-rh-frontalpole",
    "ctx-rh-temporalpole", "ctx-rh-transversetemporal", "ctx-rh-insula"
]



########## STATS ##########################

import pandas as pd

# Load your CSV
df = pd.read_csv("top_connections_by_patient_MASKED.csv")

# Create unified connection name (e.g., A ↔ B, same as B ↔ A)
df["Connection"] = df.apply(lambda row: " ↔ ".join(sorted([row["Region_1"], row["Region_2"]])), axis=1)

# Group and summarize
connection_stats = df.groupby("Connection").agg(
    Count=("Connection", "count"),
    Mean_Delta=("Delta", "mean"),
    Std_Delta=("Delta", "std")
).reset_index()

# Sort by frequency and impact
connection_stats = connection_stats.sort_values(by=["Count", "Mean_Delta"], ascending=False)


################### FUNCIONES AUXILIARES ###################

# Divide la conexión en dos líneas sin flechitas
def split_connection_simple(conn_str):
    r1, r2 = conn_str.split(" ↔ ")
    return f"{r1}\n{r2}"


################## 10 most FREQUENT CONNECTIONS ##################

top_freq = connection_stats.head(10).copy()
top_freq["Connection"] = top_freq["Connection"].apply(split_connection_simple)

plt.figure(figsize=(12, 7))
plt.barh(top_freq["Connection"], top_freq["Count"], color="steelblue")
plt.xlabel("Number of Patients", fontsize=16)
plt.title("Top 10 Most Frequent Influential Connections", fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


################## 10 most INFLUENTIAL CONNECTIONS ##################

top_impact = connection_stats.sort_values(by="Mean_Delta", ascending=False).head(10).copy()
top_impact["Connection"] = top_impact["Connection"].apply(split_connection_simple)

plt.figure(figsize=(12, 7))
plt.barh(top_impact["Connection"], top_impact["Mean_Delta"], color="darkorange")
plt.xlabel("Average Δ Prediction", fontsize=14)
plt.title("Top 10 Most Influential Connections (Mean Δ)", fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()





################## 10 most FREQUENT CONNECTIONS — ONE LINE, BIG FONT ##################

top_freq = connection_stats.head(10).copy()
# No hacemos split, dejamos los nombres originales (una línea)

plt.figure(figsize=(14, 7))  # un poco más ancho para que quepan mejor
plt.barh(top_freq["Connection"], top_freq["Count"], color="steelblue")
plt.xlabel("Number of Patients", fontsize=14)
plt.title("Top 10 Most Frequent Influential Connections", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=18)  # AUMENTADO
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


################## 10 most INFLUENTIAL CONNECTIONS — ONE LINE, BIG FONT ##################

top_impact = connection_stats.sort_values(by="Mean_Delta", ascending=False).head(10).copy()

plt.figure(figsize=(14, 7))
plt.barh(top_impact["Connection"], top_impact["Mean_Delta"], color="darkorange")
plt.xlabel("Average Δ Prediction", fontsize=14)
plt.title("Top 10 Most Influential Connections (Mean Δ)", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=18)  # AUMENTADO
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()










##################  IMPOTANCE MATRICES DIFFERENT AGES  ####################

import numpy as np

# Define 5 age groups (you can adjust the bins)
age_bins = [0, 40, 50, 60, 70, 100]
age_labels = ["<40", "40-50", "50-60", "60-70", "70+"]

df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

# Initialize dict to store one matrix per group
group_matrices = {label: np.zeros((84, 84)) for label in age_labels}
group_counts = {label: 0 for label in age_labels}

# Index map from region name to matrix index
region_list = sorted(set(df["Region_1"]) | set(df["Region_2"]))
region_to_index = {region: idx for idx, region in enumerate(region_names)}  # You must have region_names defined

# Fill matrices
for _, row in df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    delta = row["Delta"]
    group = row["Age_Group"]

    if not pd.isna(group):
        group_matrices[group][i, j] += delta
        group_matrices[group][j, i] += delta  # Symmetric
        group_counts[group] += 1


##############################  PLOT  DIFFERENT AGES #########################

for label in age_labels:
    if group_counts[label] == 0:
        continue  # Skip empty groups
    avg_matrix = group_matrices[label] / group_counts[label]

    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_matrix, cmap="magma", square=True,
                xticklabels=[region_names[i] if i % 6 == 0 else "" for i in range(84)],
                yticklabels=[region_names[i] if i % 6 == 0 else "" for i in range(84)])
    plt.title(f"Edge Importance Heatmap — Age Group: {label}")
    plt.xlabel("Brain Region")
    plt.ylabel("Brain Region")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



#####################################  PLOT MEAN ##################

# Sum all deltas to get a global matrix
global_matrix = np.zeros((84, 84))

for _, row in df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    delta = row["Delta"]
    global_matrix[i, j] += delta
    global_matrix[j, i] += delta  # Symmetric

# Count how many times each edge appears (for averaging)
edge_count = np.zeros((84, 84))
for _, row in df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    edge_count[i, j] += 1
    edge_count[j, i] += 1

# Avoid division by zero
avg_global_matrix = np.divide(global_matrix, edge_count, out=np.zeros_like(global_matrix), where=edge_count != 0)

# Plot average edge importance matrix
plt.figure(figsize=(12, 10))

sns.heatmap(
    avg_global_matrix,
    cmap="magma",
    square=True,
    xticklabels=region_names,
    yticklabels=region_names,
)

plt.title("Average Edge Importance — All Patients")
plt.xlabel("Brain Region")
plt.ylabel("Brain Region")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()





#####################################  LOG + PLOT MEAN #######################

# === 1. Crear la matriz global ===
global_matrix = np.zeros((84, 84))
edge_count = np.zeros((84, 84))

for _, row in df.iterrows():
    i = region_to_index[row["Region_1"]]
    j = region_to_index[row["Region_2"]]
    delta = row["Delta"]
    global_matrix[i, j] += delta
    global_matrix[j, i] += delta
    edge_count[i, j] += 1
    edge_count[j, i] += 1

# === 2. Promediar valores ===
avg_global_matrix = np.divide(global_matrix, edge_count, out=np.zeros_like(global_matrix), where=edge_count != 0)

# === 3. Aplicar log(x + 1) ===
avg_global_matrix_log = np.log1p(avg_global_matrix)

# === 4. Graficar la matriz log-transformada ===
plt.figure(figsize=(12, 10))
sns.heatmap(
    avg_global_matrix_log,
    cmap="viridis",
    square=True,
    xticklabels=False,
    yticklabels=False
)
plt.title("(B) Average Importance of Top 10 Connections", fontsize=24)
plt.xlabel("Brain Region", fontsize=16)
plt.ylabel("Brain Region", fontsize=16)
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# === 5. (Opcional) Ver la matriz logueada como array ===
print("\nLogged average matrix (log(x+1)):")
print(avg_global_matrix_log)
