# ADRC Bimodal Brain Age Prediction Pipeline (DTI + fMRI)

This pipeline extends the AD-DECODE implementation to a **bimodal architecture** using the **ADRC dataset**. It integrates structural (DTI) and functional (fMRI) connectomes to predict brain age using a Dual-Stream GATv2 model with early fusion.

## 1. Key Differences from AD-DECODE Pipeline

###  Dual-Modal Connectomes
- **DTI** and **fMRI** connectomes used **in parallel**
- Each modality processed by its own GATv2 stream

### Early Fusion Architecture
- Node embeddings from both GATv2 branches are **pooled and concatenated**
- Global features from both modalities are also concatenated

###  Modality-Specific Graph Metrics
- **DTI**: clustering coefficient, path length, global/local efficiency
- **fMRI**: same 4 metrics computed and normalized **separately**

###  Expanded Metadata
- Demographics: `sex`, `genotype`
- Biomarkers: `AB40`, `AB42`, `AB_ratio`, `TTAU`, `PTAU181`, `NFL`, `GFAP`  (the not available ones are set to -10)
- Graph metrics: DTI + fMRI (4 metrics each)


## 2. Preprocessing 

### 2.1 Metadata
- Loaded from Excel metadata file
- Filtered to exclude `DEMENTED=1`

- Demographics and biomarkers are shared across modalities

### 2.2 Node Features
- Extracted **FA** and **Volume** values per brain region (84 ROIs)
- Final node feature tensor per subject: `[84, 2]`
- Z-scored per region (across subjects)

- Node features  are shared across modalities

### 2.3 Connectomes
- **DTI**:
  - Log(x+1) transformed
  - 95th percentile thresholding
- **fMRI**:
  - Raw values kept (preserve negative correlations)
  


### 2.4 Subject Matching
- Only subjects with:
  - Both DTI and fMRI connectomes
  - Valid regional stats and metadata


## 3. Graph Construction

### 3.1 Per Modality Graphs
- Each subject has:
  - One DTI graph
  - One fMRI graph
- PyTorch Geometric `Data` objects with:
  - Node features: `[84, 2]`
  - Edge features: connectome values
  - Target: chronological age

### 3.2 Global Features 


In this bimodal GATv2 model, global subject-level features are distributed between the DTI and fMRI branches to avoid redundancy while preserving all relevant information.

- **Demographic features** (`sex_encoded`, `genotype`) are added only to the DTI graph. Since they are shared across modalities and constant per subject, it is unnecessary to repeat them for the fMRI stream.

- **Fluid biomarkers** (`AB40`, `AB42`, `AB_ratio`, `TTAU`, `PTAU181`, `NFL`, `GFAP`) are also shared across modalities and added only to the DTI graph.

- **DTI-specific graph metrics** (`dti_Clustering`, `dti_PathLength`, `dti_GlobalEff`, `dti_LocalEff`) are computed from the structural connectome and added to the DTI graph.

- **fMRI-specific graph metrics** (`fmri_Clustering`, `fmri_PathLength`, `fmri_GlobalEff`, `fmri_LocalEff`) are computed independently from the functional connectome and added only to the fMRI graph.

This design ensures that:

- Shared subject-level information (demographics and biomarkers) is not redundantly duplicated in both streams.
- Each modality contributes its own graph-specific topology features.
- The final model receives a concatenated global feature vector with 17 elements:  
  - 2 from demographics  
  - 7 from biomarkers  
  - 4 from DTI graph metrics  
  - 4 from fMRI graph metrics  

During model training, this vector is split and processed by different MLPs, one for each feature group, before being fused with the DTI and fMRI GNN outputs to predict brain age.



## 4. Model Architecture

### DualGATv2_EarlyFusion

- Two **GATv2 branches**:
  - 4 layers, 8 heads, residual connections, batch norm
- Shared node encoder (FA + Volume)
- Fusion of pooled DTI and fMRI graph embeddings
- Four parallel MLPs for:
  - Demographics
  - Biomarkers
  - DTI graph metrics
  - fMRI graph metrics
- Final fusion MLP outputs predicted brain age


## 5. Training

- **Cross-validation**: 7-fold stratified by age (qcut bins), 10 repetitions
- **Loss**: SmoothL1Loss (Huber loss, β = 1)
- **Optimizer**: AdamW (`lr=0.002`, `weight_decay=1e-4`)
- **Scheduler**: StepLR (`step_size=20`, `gamma=0.5`)
- **Batch size**: 6
- **Early stopping**: patience = 40 epochs
- Custom `collate_fn` for batching (DTI + fMRI pairing)



## 6. Evaluation

### Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- All metrics averaged across 7 folds × 10 repetitions

### Visualizations
- Learning curves per repetition and mean ± std
- Heatmaps: raw/log/thresholded connectomes
- Predicted vs. real age scatter plots


