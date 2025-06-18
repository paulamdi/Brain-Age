# 1_ Brain Age Prediction from Cortical Columns using GCN

This project builds a graph-based machine learning pipeline to predict brain age from cortical columns.  We use mean diffusivity (MD) values,the mean for each column, construct KNN-based graphs, and train a GCN model.

---

## 1. Data Overview

- **Input**: Column mean md
- **Subjects**: Filtered to healthy + familial risk (from AD-DECODE metadata)
- **Metadata**: Risk status extracted from `AD_DECODE_data4.xlsx`

---

## 2. Preprocessing Pipeline

### 2.1 Load Columnar Data

- Each subject has a folder with MD columns per brain region (both hemispheres).
- Column means are stacked per subject → shape: `[num_columns, 21 features]`.

### 2.2 Normalize 

- Apply z-score normalization across all columns using `StandardScaler`.

 
### 2.3 Construct Graphs

- For each subject, we build a graph where:
  - **Nodes** represent cortical columns (each with a 21-length MD profile).
  - **Edges** are defined using **K-Nearest Neighbors (KNN)**:
    - For each node (column), we compute the Euclidean distance to all other columns.
    - We then connect it to its `k = 10` nearest neighbors based on similarity.
    - This creates a subject-specific graph structure reflecting **local anatomical similarity** between columns.

- The resulting graph is represented as a `PyTorch Geometric Data` object containing:
  - `x`: Node features (21 MD values per column)
  - `edge_index`: Connectivity matrix from KNN
  - `y`: Subject's age (target)

---

## 3. Model Architecture

###  GCN with Dropout, BatchNorm, and Residual Connections

The model is a 3-layer Graph Convolutional Network (GCN) designed for processing columnar microstructure graphs:

- **GCN Layers (`GCNConv`)**: Learn representations by aggregating information from neighboring nodes.
- **Batch Normalization**: Applied after each GCN layer to stabilize training and improve convergence.
- **Dropout**: A dropout layer with `p=0.5` is applied before the final layer to prevent overfitting.
- **Residual Connection**: A skip connection is added between the second and third GCN layers to enhance gradient flow.
- **Global Mean Pooling**: Aggregates all node embeddings into a single vector for final age prediction.


####  Cross-Validation Strategy

- **7-fold stratified cross-validation** using age quantile bins to ensure balanced age distribution across folds.
- **10 repetitions per fold**, each with a different random seed, to improve the robustness of performance estimates.
- **Early stopping** with a patience of 40 epochs to avoid overfitting.
- **Batch size**: 6
- **Epochs**: up to 300 per repetition.

#### Training Details

- **Optimizer**: `AdamW` with learning rate `0.002` and `weight_decay=1e-4`
- **Scheduler**: `StepLR(step_size=20, gamma=0.5)`
- **Loss function**: `SmoothL1Loss(beta=1)`, which is less sensitive to outliers compared to MSE.

Training and validation losses are tracked for each epoch and plotted across all folds and repetitions, as well as the average ± standard deviation.

####  Learning Curves

- Individual and averaged learning curves are generated to assess model convergence and generalization.
- Includes both training and validation losses.

####  Evaluation Metrics

After training, the model is evaluated on each test set using:
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**

Results are aggregated across folds and repetitions, reporting mean ± standard deviation.

####  Prediction Visualization

- A scatter plot of **predicted vs real age** is shown with a red dashed identity line.
- A summary box displays global MAE, R², and RMSE.

Evaluation is complete
---

###  Final Model Training

After cross-validation, a final model is trained on **all healthy + familial-risk subjects** using:

- **100 epochs** (no early stopping), based on the average convergence observed during validation.
- Same architecture, optimizer, scheduler, and loss function as in CV.
  
- READY TO BE APPLIED TO ALL RISKS...



# 2_ Apply pretrained model on all healthy to all risk subjects.
-Get age, predicted age, BAG and cBAG
-Plot BAG and cBAG vs chronological age
-Violin plots-> BAG and cBAG vs Risk group, genotype and E4+/E4-
-*To do statistical tests

# 3_Metrics binary
-Compute AUC, Recall, F1Score, Precission, accuracy, ROC curve
-> 0 and best youden th
