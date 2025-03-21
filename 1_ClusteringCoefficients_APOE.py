#################  IMPORT NECESSARY LIBRARIES  ################

# Importing necessary libraries
import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions



#######################CONNECTOMES###############################################

# === 2. Define Paths ===
# Path to the ZIP file containing connectome matrices
zip_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_connectome_act.zip"

# Name of the directory inside the ZIP containing connectome files
directory_inside_zip = "connectome_act/"

# Dictionary to store connectome matrices
connectomes = {}

# === 3. Load Connectome Matrices from ZIP File ===
with zipfile.ZipFile(zip_path, 'r') as z:
    for file in z.namelist():  # Loop through all files in the ZIP
        # Filter only files that start with the directory name and end with "_conn_plain.csv"
        if file.startswith(directory_inside_zip) and file.endswith("_conn_plain.csv"):
            with z.open(file) as f:  # Open the file inside the ZIP without extracting it
                df = pd.read_csv(f, header=None)  # Read CSV without headers
                
                # Extract the subject ID by removing the path and file extension
                subject_id = file.split("/")[-1].replace("_conn_plain.csv", "")

                # Store the matrix in the dictionary with the subject ID as the key
                connectomes[subject_id] = df


# === 4. Print Total Number of Loaded Connectome Matrices ===
print(f"Total connectome matrices loaded: {len(connectomes)}")

# === 5. Remove 'whitematter' Connectomes ===
# Remove entries that contain '_whitematter' in their name
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}

print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# === 6. Extract Numeric IDs from Connectome Filenames ===
# Dictionary to store connectomes with their extracted numbers
cleaned_connectomes = {}

for k, v in filtered_connectomes.items():
    match = re.search(r"S(\d+)", k)  # Find the number after "S"
    if match:
        num_id = match.group(1)  # Extract only the number
        cleaned_connectomes[num_id] = v  # Store the connectome with its clean ID

# Display some extracted IDs to verify
print("Example of extracted connectome numbers:")
for key in list(cleaned_connectomes.keys())[:10]:  # Show first 10 extracted IDs
    print(key)

print()


#####################################METADATA##################################

# === 7. Load Metadata CSV File ===
# Path to the metadata CSV file
metadata_path = "/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data_defaced.csv"

# Load metadata into a DataFrame
df_metadata = pd.read_csv(metadata_path)

# Display the first few rows to understand the structure
print("First rows of the metadata:")
print(df_metadata.head())


# === 9. Format 'DWI' Column in Metadata ===
# Replace NaN values with "0", convert to integer (removing decimals), then to string with a leading "0"
df_metadata["DWI_fixed"] = df_metadata["DWI"].fillna(0).astype(int).astype(str)
df_metadata["DWI_fixed"] = "0" + df_metadata["DWI_fixed"]

# Display some formatted DWI values for verification
print("Example of corrected DWI values:")
print(df_metadata[["DWI", "DWI_fixed"]].head(10))


#Interpretaba como missing values lineas en blanco y no las eliminaba pq habria algo
# Remove rows where all values are NaN OR where all columns contain only empty strings or spaces
df_metadata_cleaned = df_metadata.dropna(how='all').replace(r'^\s*$', float("NaN"), regex=True)

# Drop rows where DWI is missing (since it's the key identifier)
df_metadata_cleaned = df_metadata_cleaned.dropna(subset=["DWI"])

# Check how many rows remain
print(f"Rows after forced cleaning: {df_metadata_cleaned.shape[0]}")

#family missing values porque?
# Check missing values again
print("Missing values after forced cleaning:")
print(df_metadata_cleaned.isnull().sum())


# Print all column names in metadata
print("Metadata column names:\n", df_metadata_cleaned.columns.tolist())
print()



################## MATCH CONECTOMES WITH METADATA  ##########################



# Select rows from metadata where 'DWI_fixed' is in the connectome dataset
matched_metadata = df_metadata[df_metadata["DWI_fixed"].isin(cleaned_connectomes.keys())]

# Print how many matches we found
print(f"Matched connectomes with metadata: {len(matched_metadata)} out of {len(cleaned_connectomes)}")

# === 2. Create a Dictionary for Matched Connectomes and Their Metadata ===
matched_connectomes = {row["DWI_fixed"]: cleaned_connectomes[row["DWI_fixed"]] for _, row in matched_metadata.iterrows()}



# === 4. Merge Metadata with Connectome Data ===
# Convert the dictionary into a DataFrame for easier analysis
df_matched_connectomes = pd.DataFrame(matched_metadata)






#################  VISUALIZATION AND PRINTING CONNECTOME MATRIX  ################

# === 1. Get the First Matched Connectome ===
# Retrieve the first subject from the matched connectomes
first_subject = list(matched_connectomes.keys())[0]  # Get the first subject ID
first_connectome_matrix = matched_connectomes[first_subject]  # Get the first connectome matrix

# === 2. Get the Corresponding Metadata ===
# Retrieve the corresponding metadata (DWI and age) for the first subject
first_metadata = df_matched_connectomes[df_matched_connectomes["DWI_fixed"] == first_subject].iloc[0]
first_dwi = first_metadata["DWI_fixed"]
first_age = first_metadata["age"]

# === 3. Plot the Heatmap of the First Connectome Matrix ===
plt.figure(figsize=(10, 8))  # Set figure size for the heatmap
sns.heatmap(first_connectome_matrix, cmap="viridis", cbar=True, square=True, xticklabels=False, yticklabels=False)  # Generate heatmap
plt.title(f"Connectome Matrix - DWI: {first_dwi} - Age: {first_age}")  # Add title with DWI and age
plt.xlabel("Brain Regions")  # X-axis label
plt.ylabel("Brain Regions")  # Y-axis label
plt.show()  # Display the plot

# === 4. Print the Numeric Matrix of the First Connectome (Without Indices) ===
print(f"\nConnectome Matrix for DWI: {first_dwi} - Age: {first_age}")
print(first_connectome_matrix.to_numpy())  # Print the matrix values without row/column labels



#################  PLOT THE HEATMAP WITH ZEROS IN WHITE ################

# === 5. Plot the Heatmap with 0s in White ===
plt.figure(figsize=(10, 8))  # Set figure size for the heatmap

# Replace zeros with NaN to make them appear as white in the heatmap
first_connectome_matrix_no_zeros = first_connectome_matrix.replace(0, float('nan'))

# Generate heatmap with 0s appearing as white
sns.heatmap(first_connectome_matrix_no_zeros, cmap="viridis", cbar=True, square=True, xticklabels=False, yticklabels=False)  # Generate heatmap
plt.title(f"Connectome Matrix - DWI: {first_dwi} - Age: {first_age}")  # Add title with DWI and age
plt.xlabel("Brain Regions")  # X-axis label
plt.ylabel("Brain Regions")  # Y-axis label
plt.show()  # Display the plot



#####################  LOG (X+1) #######################


# === 1. Get the First Matched Connectome ===
# Retrieve the first subject from the matched connectomes
first_subject = list(matched_connectomes.keys())[0]  # Get the first subject ID
first_connectome_matrix = matched_connectomes[first_subject]  # Get the first connectome matrix

# === 2. Get the Corresponding Metadata ===
# Retrieve the corresponding metadata (DWI and age) for the first subject
first_metadata = df_matched_connectomes[df_matched_connectomes["DWI_fixed"] == first_subject].iloc[0]
first_dwi = first_metadata["DWI_fixed"]
first_age = first_metadata["age"]

# === 3. Apply Logarithm to the Matrix ===
import numpy as np

# Add a small constant (e.g., 1) to avoid taking log(0)
first_connectome_matrix_log = np.log1p(first_connectome_matrix)  # Log(x + 1)

# === 4. Plot the Heatmap of the Log-transformed Connectome Matrix ===
plt.figure(figsize=(10, 8))  # Set figure size for the heatmap
sns.heatmap(first_connectome_matrix_log, cmap="viridis", cbar=True, square=True, xticklabels=False, yticklabels=False)  # Generate heatmap
plt.title(f"Log-transformed Connectome Matrix - DWI: {first_dwi} - Age: {first_age}")  # Add title with DWI and age
plt.xlabel("Brain Regions")  # X-axis label
plt.ylabel("Brain Regions")  # Y-axis label
plt.show()  # Display the plot

# === 5. Print the Numeric Matrix of the Log-transformed Connectome (Without Indices) ===
print(f"\nLog-transformed Connectome Matrix for DWI: {first_dwi} - Age: {first_age}")
print(first_connectome_matrix_log.to_numpy())  # Print the log-transformed matrix values without row/column labels








#################  FUNCTION: THRESHOLD CONNECTOME MATRIX  ################

import numpy as np
import pandas as pd

def threshold_connectome(matrix, percentile=100):
    """
    Apply percentile-based thresholding to a connectome matrix.

    Parameters:
    - matrix (pd.DataFrame): The original connectome matrix (84x84).
    - percentile (float): The percentile threshold to keep. 
                          100 means keep all, 75 means keep top 75%, etc.

    Returns:
    - thresholded_matrix (pd.DataFrame): A new matrix with only strong connections kept.
    """
    
    # === 1. Flatten the matrix and exclude diagonal (self-connections) ===
    matrix_np = matrix.to_numpy()
    mask = ~np.eye(matrix_np.shape[0], dtype=bool)  # Mask to exclude diagonal
    values = matrix_np[mask]  # Get all off-diagonal values

    # === 2. Compute the threshold value based on percentile ===
    threshold_value = np.percentile(values, 100 - percentile)

    # === 3. Apply thresholding: keep only values >= threshold, set others to 0 ===
    thresholded_np = np.where(matrix_np >= threshold_value, matrix_np, 0)

    # === 4. Return as DataFrame with same structure ===
    thresholded_matrix = pd.DataFrame(thresholded_np, index=matrix.index, columns=matrix.columns)
    return thresholded_matrix





#################  PLOT LOG-TRANSFORMED HEATMAPS SIDE BY SIDE  ################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Select a sample connectome matrix (first subject) ===
sample_subject_id = list(matched_connectomes.keys())[0]
sample_matrix = matched_connectomes[sample_subject_id]
sample_metadata = df_matched_connectomes[df_matched_connectomes["DWI_fixed"] == sample_subject_id].iloc[0]
sample_age = sample_metadata["age"]

# === 2. Define thresholds and apply thresholding + log transform ===
threshold_levels = [100, 95, 75, 50, 30, 5]
threshold_titles = [
    "All Connections (100%)",
    "Top 95% Strongest",
    "Top 75% Strongest",
    "Top 50% Strongest",
    "Top 30% Strongest",
    "Top 5% Strongest"
]

thresholded_log_matrices = []

for level in threshold_levels:
    # Apply threshold
    thresholded = threshold_connectome(sample_matrix, percentile=level)

    # Apply log(x + 1)
    log_transformed = np.log1p(thresholded)

    # Store result
    thresholded_log_matrices.append(log_transformed)

# === 3. Plot heatmaps side by side ===
fig, axes = plt.subplots(1, 6, figsize=(25, 5))  # 1 row, 5 columns

for i, ax in enumerate(axes):
    sns.heatmap(thresholded_log_matrices[i],
                cmap="viridis",
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
                ax=ax)
    
    ax.set_title(threshold_titles[i], fontsize=10)
    ax.set_xlabel("Regions")
    ax.set_ylabel("")

# Add overall title
fig.suptitle(f"Log-Transformed Connectome Heatmaps\nSubject: {sample_subject_id} - Age: {sample_age}", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust space to fit suptitle
plt.show()


#################  FUNCTION: CONVERT CONNECTOME MATRIX TO NETWORKX GRAPH  ################

import networkx as nx

def connectome_to_graph(matrix, threshold=0):
    """
    Convert a connectome matrix into a NetworkX graph for topological analysis.

    Parameters:
    - matrix (pd.DataFrame): The original connectome matrix (84x84).
    - threshold (float): Minimum weight required to keep an edge.

    Returns:
    - G (networkx.Graph): The resulting undirected graph.
    """
    
    # === 1. Initialize an empty undirected graph ===
    G = nx.Graph()

    # === 2. Add nodes (brain regions) ===
    num_nodes = matrix.shape[0]
    G.add_nodes_from(range(num_nodes))  # Nodes are indexed 0 to 83

    # === 3. Add edges based on the matrix values ===
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Upper triangle only (no duplicates)
            weight = matrix.iloc[i, j]
            if weight > threshold:  # Apply threshold to remove weak edges
                G.add_edge(i, j, weight=weight)

    return G


#################  PLOT GRAPHS WITH DIFFERENT THRESHOLDS  ################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# === 1. Select a sample connectome ===
sample_matrix = first_connectome_matrix  # Use one subject's connectome

# === 2. Define threshold levels ===
threshold_levels = [100, 95, 75, 50, 30, 5]
threshold_titles = [
    "All Connections (100%)",
    "Top 95% Strongest",
    "Top 75% Strongest",
    "Top 50% Strongest",
    "Top 30% Strongest",
    "Top 5% Strongest"
]

# === 3. Convert matrices to graphs with different thresholds ===
graphs = []
for level in threshold_levels:
    threshold_value = np.percentile(sample_matrix.to_numpy(), 100 - level)  # Compute threshold
    G = connectome_to_graph(sample_matrix, threshold=threshold_value)  # Convert matrix to graph
    graphs.append(G)

# === 4. Plot graphs side by side ===
fig, axes = plt.subplots(1, 6, figsize=(25, 5))  # 1 row, 5 columns

for i, ax in enumerate(axes):
    G = graphs[i]
    pos = nx.spring_layout(G, seed=42)  # Force-directed layout for clarity
    
    ax.set_title(f"{threshold_titles[i]}\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    
    nx.draw(G, pos, node_size=50, edge_color="gray", alpha=0.7, with_labels=False, ax=ax)

# Add overall title
fig.suptitle("Connectome Graphs with Different Thresholds", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust space to fit suptitle
plt.show()




#################  GLOBAL CLUSTERING COEFFICIENT WITH GENOTYPE COLORS  ################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Define thresholds and plot labels ===
threshold_levels = [100, 95, 75, 50, 30, 5]
threshold_titles = [
    "All Connections (100%)",
    "Top 95% Strongest",
    "Top 75% Strongest",
    "Top 50% Strongest",
    "Top 30% Strongest",
    "Top 5% Strongest"
]

# === 2. Define color mapping for simplified genotype categories ===
genotype_colors = {
    "APOE34 44": "red",
    "APOE33": "purple",
    "APOE23": "green"
}

# === 3. Define mapping from full genotype to simplified label ===
genotype_mapping = {
    "APOE44": "APOE34 44",
    "APOE34": "APOE34 44",
    "APOE33": "APOE33",
    "APOE23": "APOE23"
}

# === 4. Compute clustering coefficients and metadata ===
clustering_results = {title: [] for title in threshold_titles}
ages = []
genotypes = []

for subject_id, matrix in matched_connectomes.items():
    subject_data = df_matched_connectomes[df_matched_connectomes["DWI_fixed"] == subject_id]
    
    if subject_data.empty:
        continue

    age = subject_data["age"].values[0]
    genotype = str(subject_data["genotype"].values[0]).strip()

    ages.append(age)
    genotypes.append(genotype)

    for i, level in enumerate(threshold_levels):
        threshold_value = np.percentile(matrix.to_numpy(), 100 - level)
        G = connectome_to_graph(matrix, threshold=threshold_value)
        clustering_coeff = nx.transitivity(G)
        clustering_results[threshold_titles[i]].append(clustering_coeff)

# === 5. Create DataFrame with results ===
df_clustering = pd.DataFrame(clustering_results)
df_clustering["Age"] = ages
df_clustering["Genotype"] = genotypes

# === 6. Clean Genotype column and map to APOE2/3/4 ===
df_clustering["Genotype"] = df_clustering["Genotype"].astype(str).str.strip()
df_clustering["Genotype"] = df_clustering["Genotype"].replace(genotype_mapping)

print("Valores únicos en Genotype después de limpieza:")
print(df_clustering["Genotype"].unique())

# === 7. Plot individual graphs per threshold ===
for title in threshold_titles:
    plt.figure(figsize=(8, 6))

    # Assign colors based on Genotype
    colors = [genotype_colors.get(gen, "gray") for gen in df_clustering["Genotype"]]

    # Scatter plot
    plt.scatter(df_clustering["Age"], df_clustering[title], c=colors, alpha=0.7)

    # Trend line
    x = df_clustering["Age"]
    y = df_clustering[title]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    plt.plot(x, trend_line, color="black", linestyle="dashed", label=f"Slope={slope:.4f}")

    plt.xlabel("Age")
    plt.ylabel("Global Clustering Coefficient")
    plt.title(f"{title}")

    # Custom legend
    for genotype, color in genotype_colors.items():
        plt.scatter([], [], c=color, label=genotype)

    plt.legend()
    plt.grid(True)
    plt.show()

# === 8. Combined plot in 2x3 layout ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, title in enumerate(threshold_titles):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    colors = [genotype_colors.get(gen, "gray") for gen in df_clustering["Genotype"]]
    ax.scatter(df_clustering["Age"], df_clustering[title], c=colors, alpha=0.7)

    x = df_clustering["Age"]
    y = df_clustering[title]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    ax.plot(x, trend_line, color="black", linestyle="dashed", label=f"Slope={slope:.4f}")

    ax.set_xlabel("Age")
    ax.set_ylabel("Clustering Coefficient")
    ax.set_title(f"{title}")
    ax.grid(True)

    for genotype, color in genotype_colors.items():
        ax.scatter([], [], c=color, label=genotype)

    ax.legend()



fig.suptitle("Global Clustering Coefficient vs Age", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()







#################  AVERAGE CLUSTERING COEFFICIENT WITH GENOTYPE COLORS  ################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Define thresholds and plot labels ===
threshold_levels = [100, 95, 75, 50, 30, 5]
threshold_titles = [
    "All Connections (100%)",
    "Top 95% Strongest",
    "Top 75% Strongest",
    "Top 50% Strongest",
    "Top 30% Strongest",
    "Top 5% Strongest"
]

# === 2. Define color mapping for simplified genotype categories ===
genotype_colors = {
    "APOE34 44": "red",
    "APOE33": "purple",
    "APOE23": "green"
}

# === 3. Define mapping from full genotype to simplified label ===
genotype_mapping = {
    "APOE44": "APOE34 44",
    "APOE34": "APOE34 44",
    "APOE33": "APOE33",
    "APOE23": "APOE23"
}

# === 4. Compute AVERAGE clustering coefficients and metadata ===
clustering_results = {title: [] for title in threshold_titles}
ages = []
genotypes = []

for subject_id, matrix in matched_connectomes.items():
    subject_data = df_matched_connectomes[df_matched_connectomes["DWI_fixed"] == subject_id]
    
    if subject_data.empty:
        continue

    age = subject_data["age"].values[0]
    genotype = str(subject_data["genotype"].values[0]).strip()

    ages.append(age)
    genotypes.append(genotype)

    for i, level in enumerate(threshold_levels):
        threshold_value = np.percentile(matrix.to_numpy(), 100 - level)
        G = connectome_to_graph(matrix, threshold=threshold_value)
        clustering_coeff = nx.average_clustering(G, weight="weight")  # ✅ AVERAGE instead of GLOBAL
        clustering_results[threshold_titles[i]].append(clustering_coeff)

# === 5. Create DataFrame with results ===
df_clustering_avg = pd.DataFrame(clustering_results)
df_clustering_avg["Age"] = ages
df_clustering_avg["Genotype"] = genotypes

# === 6. Clean Genotype column and map to simplified labels ===
df_clustering_avg["Genotype"] = df_clustering_avg["Genotype"].astype(str).str.strip()
df_clustering_avg["Genotype"] = df_clustering_avg["Genotype"].replace(genotype_mapping)

print("Valores únicos en Genotype después de limpieza (average):")
print(df_clustering_avg["Genotype"].unique())

# === 7. Plot individual graphs per threshold ===
for title in threshold_titles:
    plt.figure(figsize=(8, 6))

    # Assign colors based on Genotype
    colors = [genotype_colors.get(gen, "gray") for gen in df_clustering_avg["Genotype"]]

    # Scatter plot
    plt.scatter(df_clustering_avg["Age"], df_clustering_avg[title], c=colors, alpha=0.7)

    # Trend line
    x = df_clustering_avg["Age"]
    y = df_clustering_avg[title]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    plt.plot(x, trend_line, color="black", linestyle="dashed", label=f"Slope={slope:.4f}")

    plt.xlabel("Age")
    plt.ylabel("Average Clustering Coefficient")
    plt.title(f"{title}")

    # Custom legend
    for genotype, color in genotype_colors.items():
        plt.scatter([], [], c=color, label=genotype)

    plt.legend()
    plt.grid(True)
    plt.show()

# === 8. Combined plot in 2x3 layout ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, title in enumerate(threshold_titles):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    colors = [genotype_colors.get(gen, "gray") for gen in df_clustering_avg["Genotype"]]
    ax.scatter(df_clustering_avg["Age"], df_clustering_avg[title], c=colors, alpha=0.7)

    x = df_clustering_avg["Age"]
    y = df_clustering_avg[title]
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    ax.plot(x, trend_line, color="black", linestyle="dashed", label=f"Slope={slope:.4f}")

    ax.set_xlabel("Age")
    ax.set_ylabel("Clustering Coefficient")
    ax.set_title(f"{title}")
    ax.grid(True)

    for genotype, color in genotype_colors.items():
        ax.scatter([], [], c=color, label=genotype)

    ax.legend()



fig.suptitle("Average Clustering Coefficient vs Age", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
