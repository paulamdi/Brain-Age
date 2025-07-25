
Counts by sex, risk groups, APOE genotype.


# 1.0_MAIN MODEL ADDECODE Trained model on all healthy subjects.py
**Main model on helathy subjects evaluated using 7 fold cv with 10 repetitions per fold
Once evaluated the model is trained on all helathy subjects**

This repository contains the full pipeline for preprocessing, training, and evaluation of a Graph Attention Network (GATv2) model to predict brain age using the AD-DECODE dataset. The pipeline processes connectomes, node/global features, and trains a GNN using 7-fold stratified cross-validation with 10 repetitions per fold.

### Overview
- **Data**: Structural connectomes (84×84), regional node features (FA, MD, Volume), demographics, graph metrics, PCA gene components.
- **Model**: 4-layer GATv2 with residual connections and batch normalization. Multi-head MLPs process global features.
- **Evaluation**: 7-fold stratified cross-validation × 10 repeats, with performance metrics (MAE,RMSE, R²) 

### Data Preprocessing
Only healthy controls retained (excludes AD and MCI).

#### Connectomes
- Loaded from ZIP archive.
- White matter on name file excluded
- Log(x+1) transformation applied.
- 70% strongest connections retained (percentile thresholding).

#### Metadata
- Extracted: `sex`, `genotype`, `systolic`, `diastolic`.
- Sex and genotype label-encoded.
- Normalized using z-scores.

#### Regional Node Features
- Extracted FA, MD, Volume (from regional stats).
- Node-wise clustering coefficient added.
- All node features normalized **per node** across subjects (z-score).

#### PCA Genes
- Top 10 age-correlated PCA components selected using Spearman correlation.
- Merged by subject ID.
- Normalized with z-scoring.

#### Graph Metrics
- Computed from log-thresholded connectomes:
  - Global clustering coefficient
  - Average shortest path length
- Normalized via z-score.

### Graph Construction
- Each subject converted into a PyTorch Geometric `Data` object:
  - **Node features**: FA, MD, Volume, clustering (shape: `[84, 4]`)
  - **Edge features**: 70%-thresholded, log-transformed connectome
  - **Global features**: concatenated tensor of metadata + graph metrics + PCA (`[16]`)
  - **Target**: chronological age

### Model Architecture
- **Node encoder**: Linear(4 → 64) + ReLU + Dropout
- **GATv2 layers**: 4 layers, 8 heads, with residual connections and batch norm
- **Global features heads**:
  - Metadata (4 → 16)
  - Graph metrics (2 → 16)
  - PCA genes (10 → 32)
- **Fusion MLP**:
  - Combines GNN graph-level output + all global embeddings
  - Final output: predicted brain age (1 scalar)

### Training Configuration
- **Loss**: SmoothL1Loss (Huber loss, β = 1)
- **Optimizer**: AdamW (`lr = 0.002`, `weight_decay = 1e-4`)
- **Scheduler**: StepLR (`step_size = 20`, `gamma = 0.5`)
- **Batch size**: 6
- **Epochs**: Up to 300 with early stopping
- **Early stopping**: Patience = 40 epochs
- **CV Strategy**: Stratified 7-fold CV using age bins, 10 repeats per fold

###  Evaluation
#### Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Computed per fold and repetition

####  Visualizations
- Learning curves (per repetition and mean ± std)
- Scatter plot of predicted vs. real age

#### Saves all predictions to CSV

### Final Model
- Trained on **all healthy subjects** (no validation split)
- Fixed training: 100 epochs (based on previous early stopping analysis)
- Final model saved 



# 1.2a_Scatter1_ADDECODE.py 1.2b_Scatter1_ADDECODE.py
Use saved CSV from cross validdation to **build scatter plots** (real vs. predicted age) 
Different types, with all the repetitions, the mean...



