AD-DECODE Brain Age Prediction Pipeline

# 0_CountingSubjectsADDECODE.py
**Counts subjects** with available connectopmes, gloal features, node features, PCA genes. 
Counts by sex, risk groups, APOE genotype.


# 1.0_Trained model on all healthy subjects.py
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


# 2.0 Predict on all risks and BAG.py
This code uses the saved trained model on all healthy subjects (end of 1.0 code) to **apply it to all risks.**

- Loads and matches all data (all risks) the same as in code 1.0
- The normalizations are based on only healthy subjects to avoid data leakage, and applies this same normalization to all risk subjects.
- Uses the pretrained model on healthy subjects from code 1.0 and applies it to all subjects
  
- **Predicts brain age** for each subject (all risks)
- Computes **BAG and cBAG**
- Vizualizes **BAG and cBAG vs age** and saves them
- **Saves CSV with real age, predicted age, BAG, cBAG, metadata, cognition metrics**


# 2.1 BAG.py
Visualizes and saves BAG and cBAG vs age plots


# 2.2_Violin plots.py
**Violin plots** of BAG and cBAG grouped by:
- Risk group
- APOE genotype (e.g., APOE33, APOE44)
- APOE risk status (E4+ vs E4−)
- Sex (M vs F) 

### Statistical comparisons included on each plot:
- Kruskal-Wallis for global group differences
- Mann-Whitney U for pairwise comparisons

CSV files with full statistical test results for BAG and cBAG


# 3_AUC Prec Recall accuracy of BAG to predict clinical group and APOE.py

Outputs of BAG/cBAG as a biomarker for :
- APOE status (E4+ vs E4−)
- Risk group comparisons: ( AD vs NoRisk and NoRisk+Familial vs MCI+AD)
- Sex (Female vs Male)

Metrics: AUC (ROC), Accuracy, Recall, Precision, F1-score
Best threshold (Youden’s J or recall ≥ 0.60)

# 4.0_SHAP 
Computes **SHAP values for global features** and node features

- Uses the pretrained model on healthy subjects from code 1.0 and applies it to all subjects

### SHAP Global Feature Analysis
Wraps the model to isolate global feature contributions (demographics, metrics, PCA genes).
Computes SHAP values with DeepExplainer.
Saves results to CSV: shap_global_features_all_subjects.csv.

### SHAP Node Feature Analysis (not used)
Wraps the model to isolate node feature contributions.
Computes SHAP values for each region across all subjects.
Averages SHAP values across node features (FA, MD, Volume, clustering).

Saves:
shap_node_feature_importance.csv → per subject and region.
shap_node_importance_by_region.csv → average SHAP per brain region.

Visualizations:
Beeswarm plot of top 20 most important brain regions.


# 4.1_SHAP global plots.py
**Builds beeswarm plots and personalized plots for global features**

Loads SHAP values for global features per subject: shap_global_features_all_subjects.csv (from code 4.0)
Loads original (real) feature values and metadata: brain_age_predictions_with_metadata.csv

*Beeswarms where each dot is colored by the real feature value, not SHAP value*
BEESWARMS:
- ONE FOR ALL global features
- One per kind of global feature
- BY AGE GROUPS (check dot color)
- BY AGE GROUP AND TYPE OF FEATURE (check dot color)

Personalized Feature Importance for 3 representative subjects: Young, middle, old

# 5_BAG-Cognition Regression.py
**LINEAR REGRESSIONS between Brain Age Gap (BAG) / Corrected BAG (cBAG) and cognitive performance** 

- For each cognitive metric the script creaes a plot with two panels ( left: BAG vs cog, right cBAG vs. cog). They include: Regression line, β coefficient (slope), R² (explained variance), p-value (statistical significance).
- Summary table for all cognitive metrics, with both BAG and cBAG stats.

# 6.0 and 6.1 (not used)
volume of hippocampus only

# 6.2_Loop for more regions No outliers
**LINEAR REGRESSIONS between Brain Age Gap (BAG) / Corrected BAG (cBAG) and brain volumes for all regions.**
Outliers are removed

Scatter plots with regression lines showing the relationship between:
  - Brain Age Gap (BAG) or corrected BAG (cBAG)
  - and z-scored relative volume of each brain region (ROI).
They include: Regression line, β coefficient (slope), R² (explained variance), p-value (statistical significance).

Summary table for all volumes, with both BAG and cBAG stats

# 7a_SHAP EDGES.py
**Computes SHAP values for edges**

Wraps the model to isolate edges
Computes SHAP values with GradientExplainer.
Saves one cvs per subject with edge pairs and their SHAP value

# 7b_SHAP EDGES plots.py (not used)
Uses shap edges values from 7a to create plots and beeswarm plots for most important edges (mean between subjects)


# 7c_glass brain top 10 mean.py
**Represents the 10 most important connections in a glass brain**

Load SHAP values from all subjects (from script 7a)
Computes the mean  SHAP value per edge across subjects
Selects the top 10 most important edges (highest average shap contribution)

Represents it in a Glass brain:
The glass brain visualization was generated using nilearn.plot_connectome, which overlays the top SHAP-based DTI connections on a standard MNI152 brain template. This background image is provided by Nilearn and does not require manual loading. The node coordinates were computed from the IIT atlas (IITmean_RPI_labels.nii.gz) using the corresponding region centroids in MNI space.


# 7d_glass brain top 10 mean left right inter.py
Since most important connections were from the left hemisphere, this code represents the top 10 connections from left, right and interhemispheric connections

