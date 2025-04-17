# ADDECODE

This folder contains scripts for working with the AD-DECODE dataset

## Scripts

### 1_Uploading ADDECODE Data.py

This script loads brain connectivity matrices (connectomes) from compressed files, processes the associated metadata, and filters out non-healthy subjects.

- Loads connectome data (`.csv`) from a `.zip` file, skipping not needed files (e.g., white matter).
- Extracts subject IDs from filenames and standardizes them.
- Loads clinical metadata from a `.csv` file and processes the `DWI` field to create unique subject identifiers.
- Matches connectome matrices with metadata based on subject IDs.
- Filters out rows with missing data and drops Alzheimer’s Disease (AD) and Mild Cognitive Impairment (MCI) subjects, keeping only healthy controls.
- Displays a sample connectome and visualizes it using a heatmap.

### GATv2_1.py

This script performs brain age prediction using Graph Attention Networks (GATv2). It integrates multimodal node features (FA, MD, Volume) with clinical and structural graph metrics.

- **Preprocessing:**
  - Normalizes connectomes and applies log + threshold transformations.
  - Extracts multimodal node features (FA, MD, Vol).
  - Encodes demographic features (e.g., sex, genotype) and combines them with graph metrics.
  
- **Graph construction:**
  - Converts each subject's connectome into a PyTorch Geometric `Data` object.
  - Includes edge features, node features(FA, MD, Vol)
  
- **Model:**
  - Defines a deep GATv2 Graph Attention Network model with 4 layers and residual connections.
  - Includes Batch Normalization, ReLU activations, Dropout, Metadata Fusion...
  - Node features: regional FA, MD, and Volume (3 per node).
  - Global features: demographic info + graph metrics.
  - Graphs are concatenated with global features and passed through a fully connected MLP.
    
- **Training & Evaluation:**
  - Loss: Smooth L1 (Huber loss)
  - Optimizer: AdamW with weight decay
  - Learning rate scheduler and early stopping
  - 7-fold stratified cross-validation (by age) with 10 repetitions per fold
  - Evaluation metrics: Mean Absolute Error (MAE), R²

- **Output:**
  - Saves model checkpoints.
  - Plots learning curves and predicted vs real ages.
  - Prints full performance metrics with standard deviation.
