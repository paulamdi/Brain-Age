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

# === 5. Filter Out 'whitematter' Connectomes ===
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





#####################  APPLY THRESHOLD + LOG TRANSFORM #######################

log_thresholded_connectomes = {}

for subject, matrix in matched_connectomes.items():
    # === 1. Apply 75% threshold ===
    thresholded_matrix = threshold_connectome(matrix, percentile=75)
    
    # === 2. Apply log(x + 1) ===
    log_matrix = np.log1p(thresholded_matrix)
    
    # === 3. Store matrix with same shape and index ===
    log_thresholded_connectomes[subject] = pd.DataFrame(log_matrix, index=matrix.index, columns=matrix.columns)




# Verificar que se han transformado las matrices correctamente
# Mostrar las primeras 5 matrices log-transformadas
for subject, matrix_log in list(log_thresholded_connectomes.items())[:5]:
    print(f"Log-transformed matrix for Subject {subject}:")
    print(matrix_log)
    print()  # Imprimir una línea vacía para separar



##################### MATRIX TO GRAPH #######################

import torch
import numpy as np
from torch_geometric.data import Data

# === Function to convert a log-transformed connectome matrix into a graph ===
def matrix_to_graph(matrix):
    """
    Converts a connectivity matrix into a graph representation.
    
    - Extracts only the upper triangle (excluding diagonal).
    - Creates an edge list (edge_index) and corresponding weights (edge_attr).
    - Assigns a basic feature vector to each node.

    Returns:
        edge_index: Tensor of connected node pairs.
        edge_attr: Tensor of edge weights.
        node_features: Tensor of node features.
    """
    indices = np.triu_indices(84, k=1)  # Extract indices of the upper triangle (ignoring diagonal)
    edge_index = torch.tensor(np.vstack(indices), dtype=torch.long)  # Stack rows to create edge list
    edge_attr = torch.tensor(matrix.values[indices], dtype=torch.float)  # Get weights for these edges
    
    node_features = torch.ones((84, 1))  # Basic feature vector for nodes (all ones) # Each node has a single feature (1)

    return edge_index, edge_attr, node_features

########################### CONVERT MATRICES TO GRAPHS ##########################

# Create a dictionary to store the converted graphs (log-transformed matrices)
graph_data_list = []

# Iterate over the matched connectomes with log-transformed matrices
for subject, matrix_log in log_thresholded_connectomes.items():
    # Convert the log-transformed matrix to graph format
    edge_index, edge_attr, node_features = matrix_to_graph(pd.DataFrame(matrix_log))  # Convert to graph

    # Retrieve the corresponding metadata (age, etc.) for this subject
    age_row = df_metadata.loc[df_metadata["DWI_fixed"] == subject, "age"]
    if not age_row.empty:
        age = torch.tensor([age_row.values[0]], dtype=torch.float)  # Age as a tensor
        
        # Create the graph data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=age)
        
        # Append the graph data to the list
        graph_data_list.append(data)

# Display the first graph's data structure for verification
print(f"Example graph structure: {graph_data_list[0]}")


##################### PLOT GRAPH ########################

import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(edge_index):
    G = nx.Graph()
    G.add_edges_from(edge_index.T.numpy())  # Convert tensor to list of edges

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=100, alpha=0.2, edge_color="gray")
    plt.title("Graph Representation of a Connectome")
    plt.show()

plot_graph(edge_index)





######################  DEFINE MODEL #########################

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class BrainAgeGNN(torch.nn.Module):
    def __init__(self):
        super(BrainAgeGNN, self).__init__()

        # Graph Convolutional Layers
        self.conv1 = GCNConv(1, 64)
        self.bn1 = BatchNorm(64)

        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)

        self.conv3 = GCNConv(128, 128)
        self.bn3 = BatchNorm(128)

        # Fully Connected Layer for Age Prediction
        self.fc = torch.nn.Linear(128, 1)
        
        # Dropout for Regularization
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GCN Layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)

        # Second GCN Layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # Save skip connection after second layer
        x_residual = x  # (Batch, 128) → Matches conv3 output

        # Third GCN Layer
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        # Skip Connection: Add output from conv2 to conv3
        x = x + x_residual  # Both tensors have (Batch, 128)

        # Global Mean Pooling (Summarizes graph into single vector)
        x = global_mean_pool(x, data.batch)

        # Dropout before the final fully connected layer
        x = self.dropout(x)

        # Fully Connected Layer for Age Prediction
        x = self.fc(x)

        return x
    
    
from torch.optim import Adam
from torch_geometric.loader import DataLoader  # Usamos el DataLoader de torch_geometric

# === Training function ===
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data).view(-1)  # Predicción de edad
        loss = criterion(output, data.y)  # MSE Loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# === Evaluate function ===
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data).view(-1)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader)




import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

# Training parameters
epochs = 100
patience = 20  # Early stopping

k = 8  # Folds
batch_size = 8

# === Initialize losses ===
all_train_losses = []
all_test_losses = []

#  K-Fold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(graph_data_list)):
    print(f'\n--- Fold {fold+1}/{k} ---')

    # Train and test data
    train_data = [graph_data_list[i] for i in train_idx]
    test_data = [graph_data_list[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    
    model = BrainAgeGNN()
    
    
    # Adam optimizer is used to adjust model parameters during training
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    # Agregar un Scheduler para reducir el learning rate cada 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    
    # Smooth L1 Loss is used instead of MSE to reduce sensitivity to outliers
    criterion = torch.nn.SmoothL1Loss(beta=1)  # Controls the transition between L1 and L2 loss



    # Save best model
    best_model_path = f"model_fold_{fold+1}.pt"
    best_loss = float('inf')
    patience_counter = 0

    # Save losses in each epoch
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

        #  Guardar modelo si es el mejor hasta ahora
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)  # GUARDAMOS EL MEJOR MODELO
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break  # Salimos del entrenamiento si no mejora
                
        scheduler.step()
        print(f"Epoch {epoch+1}: Current Learning Rate: {optimizer.param_groups[0]['lr']}")
       

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)
    
    



# === Learning curve graph ===
plt.figure(figsize=(10, 6))

for fold in range(k):
    plt.plot(all_train_losses[fold], label=f'Train Loss - Fold {fold+1}', linestyle='dashed')
    plt.plot(all_test_losses[fold], label=f'Test Loss - Fold {fold+1}')

plt.xlabel("Epochs")
plt.ylabel("Smooth L1 Loss")
plt.title("Learning curve")
plt.legend()
plt.grid()
plt.show()


##############PREDICTIONS#######################

from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# === Lists to store real and predcted values===

y_true = []
y_pred = []

for fold, (train_idx, test_idx) in enumerate(kf.split(graph_data_list)):
    print(f'\n--- Evaluating Fold {fold+1}/{k} ---')

    # Test data
    test_data = [graph_data_list[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Upload model from training
    model = BrainAgeGNN()
    model.load_state_dict(torch.load(f"model_fold_{fold+1}.pt"))  # CARGAMOS LOS PESOS DEL MEJOR MODELO
    model.eval()  # Evaluation mode

    with torch.no_grad():
        for data in test_loader:
            output = model(data).view(-1)  # Predicciones d
            y_pred.extend(output.tolist())  # Save predicciones
            y_true.extend(data.y.tolist())  # Save real values



# Lists to arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Calculate MAE and  R² ===
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)



# === Graph real vd predicted values===
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label="Predictions")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="dashed", label="Ideal (y=x)")


textstr = f"MAE: {mae:.2f}\nR²: {r2:.2f}"
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
plt.title("Real Ages vs Predicted Ages")
plt.legend(loc="upper left")  
plt.grid(True)
plt.show()



############### PREDICT SOME AGES #################

import random

# Seleccionamos 5 ejemplos aleatorios del conjunto de prueba
random_indices = random.sample(range(len(y_true)), 5)

print("\n Example predictions:")
for idx in random_indices:
    print(f"Pacient {idx+1}: Real Age = {y_true[idx]:.1f} years | Prediction = {y_pred[idx]:.1f} years")
