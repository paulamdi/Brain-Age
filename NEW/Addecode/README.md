### `1_Trained model on all healthy subjects.py`

This script performs the full preprocessing and training pipeline for the **brain age prediction model** using only **healthy subjects** from the AD-DECODE dataset. It includes:

 **Data Preprocessing**
- Loads structural connectomes (84×84 matrices) from zip files.
- Filters out non-usable subjects (e.g., with AD/MCI or missing connectomes).
- Loads and matches subject-level metadata (age, sex, genotype, blood pressure, etc.) 
- Computes **graph metrics** (clustering coefficient, path length) added as global features
- Incorporates multimodal brain features: **FA, MD, and Volume** for 84 regions added as node features
- Node clustering coefficients added as node feature
  
 **Genetics Integration**
- Uses top 10 age-correlated gene expression **PCA components** (from blood RNA) added as global features
- Features are z-scored using healthy controls.

 **Graph Construction**
- Each subject becomes a PyTorch Geometric `Data` object with:
  - Node features (FA, MD, Vol, clustering)
  - Edge weights (log-transformed and thresholded connectomes)
  - Global features (metadata + PCA)
  - Age as regression target

 **Model Architecture**
- A 4-layer **GATv2** Graph Neural Network with:
  - Edges
  - Multi-head attention (8 heads)
  - Residual connections between layers
  - Batch normalization
  - Node embedding layer
- **Multi-branch fusion** for:
  - Demographics (4 vars)
  - Graph metrics (2 vars)
  - Gene PCA features (10 PCs)
- Final MLP combines GNN output with metadata branches.

 **Cross-Validation & Training**
- 7-fold **Stratified K-Fold** CV (based on age bins), 10 repeats per fold.
- Each fold trains for up to 300 epochs with early stopping (patience = 40).
- Loss: **Smooth L1 Loss (Huber)**. learning scheduler
- Tracks train/test loss per epoch and plots learning curves (mean ± std).

 **Evaluation**
- Computes **MAE and R²** across all folds and repeats.
- Plots:
  - Learning curves
  - Predicted vs Real Age scatter
- Saves the final metrics and trained models.

 **Final Model**
- After evaluation, retrains on **all healthy subjects** (100 epochs) and saves as: model_trained_on_all_healthy.pt


### `2_Predict on all risks and BAG.py`

This script uses the trained brain age model (from healthy subjects) to predict **brain age**, **BAG (Brain Age Gap)**, and **cBAG (corrected BAG)** across the full AD-DECODE dataset, including **controls, familial risk, MCI, and AD subjects**.

 **Preprocessing (repeated)**
- Reloads and preprocesses:
  - Connectomes (log-thresholded at 95%)
  - Regional FA, MD, Volume features
  - Graph metrics: clustering coefficient and path length
  - Metadata and gene PCA (top 10 age-correlated components)
- Features are **z-scored using healthy control statistics** to ensure consistent scaling.

 **Graph Construction**
- Converts each subject into a PyTorch Geometric `Data` object with:
  - Node features (FA, MD, Vol, clustering)
  - Global metadata (demographics, graph metrics, PCA)
  - Ground-truth age
  - Subject ID for tracking

 **Brain Age Prediction**
- Loads the pretrained model from `model_trained_on_all_healthy.pt`
- Predicts **brain age** for every subject
- Computes:
  - **BAG** = Predicted Age − Chronological Age
  - **cBAG** = BAG corrected via linear regression on age (removes bias)

 **Data Export**
- Saves predictions, BAG/cBAG, and full metadata to:
  - `brain_age_predictions_with_metadata.csv`(metadata normalized)
  - `BAG_with_all_metadata.csv` 

 **Visualizations**
- **Scatter plots**: BAG and cBAG vs. Age (pre- and post-correction)
- **Violin plots**:
  - By **Risk group** (`NoRisk`, `Familial`, `MCI`, `AD`)
  - By **Genotype** (APOE23, 33, 34, 44)
  - By **APOE risk status** (E4− vs. E4+)
  - By **Sex**

 **Statistical Tests**
- For each group variable (Risk, Genotype, APOE, Sex) and outcome (BAG, cBAG):
  - Global **Kruskal-Wallis** test
  - All pairwise **Mann-Whitney U** tests
- Significance encoded with: `ns`, `*`, `**`, `***`, `****`
- Results saved to:
  - `stat_results_BAG.csv`
  - `stat_results_cBAG.csv`

 This script evaluates how brain aging (BAG) varies across clinical and genetic risk factors.



### `3_AUC Prec Recall accuracy of BAG to predict ....py`

This script evaluates the discriminative power of **Brain Age Gap (BAG)** and **corrected BAG (cBAG)** as biomarkers using **ROC curves**, **AUC**, **thresholding**, and **classification metrics** across several biological and clinical variables.

 **Main Goals**
- Assess whether BAG/cBAG can classify:
  - **APOE E4+ vs E4−**
  - **AD vs NoRisk**
  - **NoRisk+Familial vs MCI+AD**
  - **Sex (Female vs Male)**

 **Metrics Computed**
For each comparison:
- ROC Curve & AUC
- Accuracy, Recall (Sensitivity), Precision, F1-score
- Confusion Matrix
- Violin plot of BAG/cBAG distributions

 **Comparisons performed**
- **APOE E4+ prediction** using BAG and cBAG:
  - Standard and **recall-prioritized** evaluations
  - Visual ROC curves and violin plots per genotype
- **RISK classification**:
  - **AD vs NoRisk**
  - **NoRisk+Familial vs MCI+AD**
  - Results include ROC, AUC, and violin plots
- **Sex classification**:
  - Female vs Male, using both BAG and cBAG
  - Separate analysis for full sample and healthy-only subset

 **Threshold analysis**
- Standard: best threshold based on **Youden’s J**
- Priority mode: selects **lowest threshold reaching recall ≥ 0.60**, to favor sensitivity

 **Outputs**
- ROC curves for each task
- Confusion matrices for classification performance
- Violin plots visualizing the BAG/cBAG distribution across target groups
- Full classification reports printed for each setting

 This script validates **BAG/cBAG as candidate biomarkers**, especially for genetic risk (APOE), diagnostic category (AD, MCI), and sex-based stratification in the AD-DECODE cohort.



### `4.0_SHAP.py`

This script performs SHAP analysis on a pretrained BrainAge GATv2 model to interpret the contribution of global and node-level features to the brain age prediction in the AD-DECODE dataset.

Key functionalities:

1. **Data Preparation**
   - Loads connectomes, demographic data, PCA gene features, and multimodal node features (FA, MD, Volume).
   - Normalizes all features using statistics from healthy controls only.

2. **Model Loading**
   - Loads a pretrained graph neural network (GATv2) trained on healthy subjects to predict age.
   - Wraps the model for SHAP analysis, isolating either global or node-level features.

3. **SHAP for Global Features**
   - Computes SHAP values using `shap.DeepExplainer` on 16 global features (demographics, connectome metrics, and PCA gene components).
   - Saves SHAP values to CSV for all subjects.

4. **Visualization**
   - Beeswarm plots for:
     - All global features combined
     - Grouped by feature type (demographics, graph metrics, PCA)
     - Stratified by age (young, middle, old)
     - Grouped by age and feature type
     - Per individual subject (example from each age group)

5. **SHAP for Node Features**
   - Uses `shap.GradientExplainer` to analyze 84 brain regions' contribution to age prediction.
   - Computes region-wise average SHAP importance across all subjects.
   - Saves signed SHAP values per brain region and per subject.

6. **Beeswarm for Node Regions**
   - Generates beeswarm plots of the most important brain regions based on node-level SHAP values.

This script provides insight into which features and brain regions are most predictive of brain age, supporting interpretability of the GNN model.



### `4.1_SHAP_EDGES.py`

This script applies SHAP analysis to **connectome edge weights** in the AD-DECODE dataset to understand which **brain connections** most influence brain age predictions.


**1. Data & Model Setup**
- Loads preprocessed connectomes, node features (FA, MD, Volume, clustering), and global features (demographics, graph metrics, PCA genes).
- Normalizes features using **healthy controls only**.
- Loads the **pretrained GATv2 brain age model** trained on healthy subjects.


**2. SHAP Edge Analysis**
- Uses `shap.GradientExplainer` on `edge_attr` (connection strengths).
- For each subject, computes SHAP values for every brain connection.
- Saves SHAP scores per edge as:  
  `edge_shap_subject_<ID>.csv`


**3. Visualizations**
- **Top 10 edges** per subject: horizontal barplots (e.g., young, middle, old).
- **Beeswarm plot** for top 10 most influential edges across all subjects.

Reveal which **brain connections** drive the model's age predictions, at both **individual and population levels**.



### `4.2_SHAP_EDGES_beeswarm.py`

Generates **beeswarm plots** of SHAP values for brain connections (edges), showing their contribution to brain age predictions.

**Plots Included:**

- **Beeswarm (top 10 edges) colored by age group**  
  → Young (<45), Middle (45–64), Old (≥65)

- **Beeswarm per age group**  
  → Separate plots for Young, Middle, and Old

- **Beeswarm (top 10 edges) colored by clinical risk group**  
  → NoRisk, Familial, MCI, AD

- **Beeswarm per risk group**  
  → Separate plots per group

All edges are labeled using brain region names, and ranked by average SHAP importance.


### `5_BAG_cognition_regression.py`

Performs **linear regressions** between Brain Age Gap (BAG / cBAG) and **cognitive performance metrics**.

**Key steps:**
- Loads `BAG_with_all_metadata.csv`
- Loops through ~30 cognitive scores
- Regresses each against:
  - **BAG**
  - **Corrected BAG (cBAG)**
- Collects R², p-values, and beta coefficients

**Outputs:**
- **Scatter plots** with regression line and p-value:
  - Side-by-side for BAG and cBAG
- Saves summary results to:
  - `regression_results_BAG_cognition.csv`



### 6.0 BAG vs. Hippocampal Volume 

This script investigates the relationship between Brain Age Gap (BAG) and hippocampal volume using regional structural MRI data.


The script generates **4 regression plots**:

1. Left Hippocampus (z-scored relative volume) ~ BAG  
2. Right Hippocampus (z-scored relative volume) ~ BAG  
3. Left Hippocampus (z-scored relative volume) ~ cBAG  
4. Right Hippocampus (z-scored relative volume) ~ cBAG

Each plot includes:
- One dot per subject
- A linear regression line
- Annotated values for:
  - R² (explained variance)
  - p-value
  - β (regression slope)

These plots visually test whether accelerated brain aging (BAG > 0) is associated with lower hippocampal volume.

Additional output
A CSV file `regression_BAG_vs_Rel_z_hippocampus.csv` with regression stats (R², p, β) for each comparison.


### 6.1 BAG vs. Hippocampal Volume (outlier-robust version)
This script replicates the analysis described in Section 6.0 but includes automatic outlier removal to improve regression robustness.

It investigates the relationship between Brain Age Gap (BAG / cBAG) and z-scored relative hippocampal volume, using structural MRI data.

Main difference:
Subjects with extreme values are excluded before regression:

|BAG| > 20

|Hippocampal volume z-score| > 3

This helps reduce the influence of outliers on statistical results.

### 6.2 BAG vs volume loop
The same as previous script but uses a loop to be able to anlyse more brain regions



