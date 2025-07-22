# AD-DECODE Brain Age Prediction Pipeline

## 0_CountingSubjectsADDECODE.py
Counts subjects with available connectopmes, gloal features, node features, PCA genes. 
Counts by sex, risk groups, APOE genotype.


## 1.0_Trained model on all healthy subjects.py
This repository contains the full pipeline for preprocessing, training, and evaluation of a Graph Attention Network (GATv2) model to predict brain age using the AD-DECODE dataset. The pipeline processes connectomes, node/global features, and trains a GNN using 7-fold stratified cross-validation with 10 repetitions per fold.

### 1. Overview
- **Data**: Structural connectomes (84×84), regional node features (FA, MD, Volume), demographics, graph metrics, PCA gene components.
- **Model**: 4-layer GATv2 with residual connections and batch normalization. Multi-head MLPs process global features.
- **Evaluation**: 7-fold stratified cross-validation × 10 repeats, with performance metrics (MAE,RMSE, R²) 

### 2. Data Preprocessing
Only healthy controls retained (excludes AD and MCI).

#### 2.1 Connectomes
- Loaded from ZIP archive.
- White matter on name file excluded
- Log(x+1) transformation applied.
- 70% strongest connections retained (percentile thresholding).

#### 2.2 Metadata
- Extracted: `sex`, `genotype`, `systolic`, `diastolic`.
- Sex and genotype label-encoded.
- Normalized using z-scores.

#### 2.3 Regional Node Features
- Extracted FA, MD, Volume (from regional stats).
- Node-wise clustering coefficient added.
- All node features normalized **per node** across subjects (z-score).

#### 2.4 PCA Genes
- Top 10 age-correlated PCA components selected using Spearman correlation.
- Merged by subject ID.
- Normalized with z-scoring.

#### 2.5 Graph Metrics
- Computed from log-thresholded connectomes:
  - Global clustering coefficient
  - Average shortest path length
- Normalized via z-score.

### 3. Graph Construction
- Each subject converted into a PyTorch Geometric `Data` object:
  - **Node features**: FA, MD, Volume, clustering (shape: `[84, 4]`)
  - **Edge features**: 70%-thresholded, log-transformed connectome
  - **Global features**: concatenated tensor of metadata + graph metrics + PCA (`[16]`)
  - **Target**: chronological age

### 4. Model Architecture
- **Node encoder**: Linear(4 → 64) + ReLU + Dropout
- **GATv2 layers**: 4 layers, 8 heads, with residual connections and batch norm
- **Global features heads**:
  - Metadata (4 → 16)
  - Graph metrics (2 → 16)
  - PCA genes (10 → 32)
- **Fusion MLP**:
  - Combines GNN graph-level output + all global embeddings
  - Final output: predicted brain age (1 scalar)

### 5. Training Configuration
- **Loss**: SmoothL1Loss (Huber loss, β = 1)
- **Optimizer**: AdamW (`lr = 0.002`, `weight_decay = 1e-4`)
- **Scheduler**: StepLR (`step_size = 20`, `gamma = 0.5`)
- **Batch size**: 6
- **Epochs**: Up to 300 with early stopping
- **Early stopping**: Patience = 40 epochs
- **CV Strategy**: Stratified 7-fold CV using age bins, 10 repeats per fold

### 6. Evaluation
#### 6.1 Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Computed per fold and repetition

#### 6.2 Visualizations
- Learning curves (per repetition and mean ± std)
- Scatter plot of predicted vs. real age

### 7. Final Model
- Trained on **all healthy subjects** (no validation split)
- Fixed training: 100 epochs (based on previous early stopping analysis)
- Final model saved 



