# SHAP-CL 

The SHAP-guided contrastive learning pipeline is organized into scripts, each corresponding to a step in the process:

###  `1_MAIN MODEL ADDECODE Trained model on all healthy subjects.py` - SAME ADDECODE MODEL
- Trains the GATv2 model using only healthy control subjects from the AD-DECODE dataset (explained in ADDECODE Baseline repository).
- Saves the trained model to be used for SHAP computation.

###  `7.0 SHAP cvs gobal features.py`
- Applies the trained model to all subjects (healthy + at-risk) to compute SHAP values for global features.
- **SHAP values are stored in a CSV file**

- **Wrapper for SHAP**: A `GlobalOnlyModel` wrapper is used to isolate the global features branch (e.g., demographics, graph metrics, PCA genes).

- **SHAP computation**:
  - SHAP values are computed for all global features of each subject.
  - Results are stored in a list along with subject metadata (ID, diagnosis, sex, genotype, etc.).
  -  Saves a CSV file with per-subject SHAP values and metadata:


### `7.1c similarity triplets top k.py`
- Loads the SHAP CSV file and z-score normalizes SHAP vectors.
- Computes **cosine similarity** between all subject pairs.
- Constructs **triplets using the top-k strategy**:
  - For each anchor, selects 5 most similar (positive) and 5 least similar (negative) SHAP profiles.

### `7.2_contrative learning.py`
- Trains a **contrastive embedding model** (2-layer MLP) using the triplets generated in step 7.1c .
- Optimizes a contrastive loss (NT-Xent) to project SHAP vectors into a latent embedding space.
- Saves the trained projection head for later embedding extraction.

### `7.3.1_UMAP visualization with stadard scaler.py`
- Applies the trained SHAP encoder to all subjects.
- Uses UMAP for dimensionality reduction and visualizes the 2D projection of SHAP embeddings.
- Colors points by age, risk group, or other metadata for interpretation.


