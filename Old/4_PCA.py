# === 1. Import Necessary Libraries ===
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

# === 4. Display a Few Connectome Matrices ===
for subject, matrix in list(connectomes.items())[:5]:  # Iterate over the first 5 loaded matrices
    plt.figure(figsize=(10, 8))  # Set figure size
    sns.heatmap(matrix, cmap="viridis", cbar=True, square=True, xticklabels=1, yticklabels=1)  # Generate heatmap with "viridis" color map
    plt.title(f"Connectome Matrix - {subject}")  # Add title with subject ID
    plt.xlabel("Brain Regions")  # X-axis label
    plt.ylabel("Brain Regions")  # Y-axis label
    plt.show()  # Display the plot

# === 5. Print Total Number of Loaded Connectome Matrices ===
print(f"Total connectome matrices loaded: {len(connectomes)}")

# === 6. Filter Out 'whitematter' Connectomes ===
# Remove entries that contain '_whitematter' in their name
filtered_connectomes = {k: v for k, v in connectomes.items() if "_whitematter" not in k}

print(f"Total connectomes after filtering: {len(filtered_connectomes)}")

# === 8. Extract Numeric IDs from Connectome Filenames ===
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






################## MATCH CONECTOMES WITH METADATA  ##########################



# Select rows from metadata where 'DWI_fixed' is in the connectome dataset
matched_metadata = df_metadata[df_metadata["DWI_fixed"].isin(cleaned_connectomes.keys())]

# Print how many matches we found
print(f"Matched connectomes with metadata: {len(matched_metadata)} out of {len(cleaned_connectomes)}")

# === 2. Create a Dictionary for Matched Connectomes and Their Metadata ===
matched_connectomes = {row["DWI_fixed"]: cleaned_connectomes[row["DWI_fixed"]] for _, row in matched_metadata.iterrows()}

# === 3. Verify Matching by Printing Some Examples ===
print("\nExample of matched connectomes and metadata:")
for key in list(matched_connectomes.keys())[:10]:  # Print first 10 matches
    print(f"Connectome ID: {key} -> Matched with metadata row")

# === 4. Merge Metadata with Connectome Data ===
# Convert the dictionary into a DataFrame for easier analysis
df_matched_connectomes = pd.DataFrame(matched_metadata)

# Print the first few rows of the merged dataset
print("\nPreview of matched metadata:")
print(df_matched_connectomes.head())

# DATA ANALYSIS

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


# === UPPER TRIANGLE ===
import numpy as np
import pandas as pd
import os

# Function to extract the upper triangle (excluding the diagonal)
def get_upper_triangle(matrix):
    return matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))

# Apply this function to all connectomes and store them
connectomes_upper = {subject: get_upper_triangle(matrix) for subject, matrix in cleaned_connectomes.items()}

# Save each processed connectome as a CSV file
output_dir = "/home/bas/Desktop/MyData/Processed_Connectomes/"  # Change this path if needed
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

for subject, matrix in connectomes_upper.items():
    output_path = os.path.join(output_dir, f"{subject}_upper_triangle.csv")
    matrix.to_csv(output_path, index=False, header=False)  # Save without headers and index

print(f"Processed {len(connectomes_upper)} connectomes and saved in {output_dir}")



###########################  PCA  #################################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Convert each connectome (84x84) into a vector of the upper triangle (3486 connections)
connectome_vectors = []
subject_ids = []

for subject, matrix in connectomes_upper.items():
    # Upper triangle withput diagonal
    vector = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool)).stack().values
    connectome_vectors.append(vector)
    subject_ids.append(subject)

#  DataFrame
connectome_vectors = np.array(connectome_vectors)
df_connectomes = pd.DataFrame(connectome_vectors, index=subject_ids)


from sklearn.feature_selection import VarianceThreshold

# Remove features with very low variance
selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
filtered_data = selector.fit_transform(df_connectomes)



# === 2. PCA ===
pca = PCA(n_components=0.9)  #  90%
principal_components = pca.fit_transform(filtered_data)


explained_variance = np.cumsum(pca.explained_variance_ratio_)

# === 3. Plot PCA Explained Variance ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

# === 4. Save data
df_pca = pd.DataFrame(principal_components, index=subject_ids)

print(f"Dimensiones después de PCA: {df_pca.shape}")  # Verify dimensions

# === 5. Update to use PCA in the gnn ===
connectome_vectors_pca = df_pca.values

########################### TRANSFORM TO GRAPHS #############################

import torch
from torch_geometric.data import Data

# === Graph dataset with PCA ===
data_list = []  # List

for i, subject in enumerate(subject_ids):
    # Reduced features
    node_features = torch.tensor(connectome_vectors_pca[i], dtype=torch.float).unsqueeze(1)  # Agregamos dimensión extra
    
    # Graph with PCA features
    num_nodes = node_features.shape[0]
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T  # Grafo completo
    edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)  # Pesos arbitrarios

    # Age and drop Nan
    age_row = df_metadata.loc[df_metadata["DWI_fixed"] == subject, "age"].dropna()
    
    if not age_row.empty:
        age = torch.tensor([age_row.values[0]], dtype=torch.float)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=age)
        data_list.append(data)

print(f"Created dataset with {len(data_list)} samples using PCA.")




########################### VISUALIZE A GRAPH #############################

import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(data):
    G = nx.Graph()
    G.add_edges_from(data.edge_index.T.numpy())  # Convert tensor to list of edges

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=100, alpha=0.2, edge_color="gray")
    plt.title("Graph Representation of a Connectome")
    plt.show()

# Example
if data_list:
    plot_graph(data_list[0])


################# TRAINING ###############################


#DEFINE THE GNN

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

import torch.nn.functional as F

class BrainAgeGNN(torch.nn.Module):
    def __init__(self):
        super(BrainAgeGNN, self).__init__()

        self.conv1 = GCNConv(1, 64)
        self.bn1 = BatchNorm(64)

        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)

        self.conv3 = GCNConv(128, 128)
        self.bn3 = BatchNorm(128)

        self.fc = torch.nn.Linear(128, 1)  

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        x_residual = x

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x = x + x_residual

        x = global_mean_pool(x, data.batch)

        x = self.dropout(x)

        x = self.fc(x)

       

        return x

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import random
import os

# Create directory to save models
os.makedirs("saved_models", exist_ok=True)

# === Training and Evaluation Functions ===

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data).view(-1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# === General Configuration ===
epochs = 500
patience = 20
learning_rate = 0.0001
k = 6  # number of folds
batch_size = 8

all_train_losses = []
all_test_losses = []

# K-Fold Cross Validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
    print(f'\n--- Fold {fold+1}/{k} ---')

    # Prepare train and test sets
    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, loss function
    model = BrainAgeGNN()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

        # Early stopping and model saving
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0

            # Save best model
            best_model_path = f"saved_models/model_fold_{fold+1}.pt"
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)

# === Learning Curve Plot ===
plt.figure(figsize=(10, 6))
for fold in range(k):
    plt.plot(all_train_losses[fold], label=f'Train Loss - Fold {fold+1}', linestyle='dashed')
    plt.plot(all_test_losses[fold], label=f'Test Loss - Fold {fold+1}')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Learning Curve - K-Fold Cross Validation")
plt.legend()
plt.grid()
plt.show()


##############  FINAL EVALUATION & PREDICTIONS  #######################

y_true = []
y_pred = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    test_data = [data_list[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load saved model
    model = BrainAgeGNN()
    model.load_state_dict(torch.load(f"saved_models/model_fold_{fold+1}.pt"))
    model.eval()

    # Generate predictions
    with torch.no_grad():
        for data in test_loader:
            output = model(data).view(-1)
            y_pred.extend(output.tolist())
            y_true.extend(data.y.tolist())

# Convert predictions and true values to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Final Metrics ===
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\n Final Results:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# === Plot: Real vs Predicted Age ===
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label="Predictions")
plt.plot([20, 100], [20, 100], color="red", linestyle="dashed", label="Ideal (y = x)")
plt.xlim(20, 100)
plt.ylim(20, 100)
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Comparison of Real vs Predicted Ages (20–100 years)")
plt.legend()
plt.grid(True)
plt.show()


############### EXAMPLE PREDICTIONS #################

# Print 10 random examples from test results
random_indices = random.sample(range(len(y_true)), 10)

print("\n Example Predictions:")
for idx in random_indices:
    print(f"Patient {idx+1}: Real Age = {y_true[idx]:.1f} years | Predicted Age = {y_pred[idx]:.1f} years")

print(f"Max predicted age: {max(y_pred):.2f}")
print(f"Min predicted age: {min(y_pred):.2f}")
