
## 0_PearsonCorrelation.py

This script computes Pearson correlations of brain connectomes across age groups using data from both AD-DECODE and ADNI datasets.

- Loads and preprocesses connectome matrices and metadata for both datasets.
- Filters to include only healthy control (CN) subjects.
- Organizes subjects into predefined age bins (20–30, 30–40, ..., 80–90).
- Computes average connectome vectors per age group.
- Calculates intra-group and cross-dataset Pearson correlations:
  - Within AD-DECODE.
  - Within ADNI.
  - Between AD-DECODE and ADNI.
  - Subject-to-subject across datasets.

## 1_GAT adni.py - ADNI Pretraining 

This script implements **pretraining on the ADNI dataset** using a Graph Attention Network (GATv2) model.  
The architecture and preprocessing steps were carefully adapted to **maximize compatibility with the AD-DECODE dataset**, facilitating better transfer learning.

Although more graph metrics (e.g., efficiency) were available in ADNI, this setup intentionally **excludes features not used in AD-DECODE**.

The ADNI model mirrors the AD-DECODE model, but **without**:
 - **Mean Diffusivity (MD)**  
 - **Blood Pressure** (Systolic, Diastolic)  
 - **Transcriptomic PCA components**

###  Features used in ADNI pretraining:

- **Node features**:
  - Fractional Anisotropy (FA)
  - Regional Volume
  - Node-wise Clustering Coefficient

- **Global features**:
  - Sex (one-hot)
  - APOE genotype (label-encoded)
  - Global Clustering Coefficient
  - Path Length


After CV, the model was retrained on **all healthy ADNI subjects** and saved 



## 2_Full Fine Tuning.py - AD-DECODE Full Fine-Tuning 

This script performs **full fine-tuning** on the AD-DECODE dataset, starting from a model **pretrained on ADNI**.

The goal is to **reuse the structural representations learned from ADNI**, while allowing the model to adapt to **new features unique to AD-DECODE**, such as Mean Diffusivity, blood pressure, and transcriptomic PCA components.



### Transfer Learning Strategy

1. **Initialization**:
   - The model is initialized with the weights from the GATv2 model trained on healthy ADNI subjects.

2. **Selective loading**:
   - **Only compatible layers are loaded** from the pretrained model.
   - The following layers are **excluded**:
     - `node_embed`: because AD-DECODE uses **3 node features** (FA, MD, Volume), while ADNI only used 2.
     - `fc.0`: the input dimensionality of the final MLP has changed (AD-DECODE includes more global features).

3. **Weight transfer**:
   - All other layers, including the 4 GATv2 layers and remaining parts of the MLP, **retain the pretrained weights** from ADNI.
   - These weights encode generalizable structural patterns related to healthy brain aging.

4. **Full Fine-Tuning (No Freezing)**:
   - **All layers are left trainable** (`unfrozen`), allowing the model to:
     - Retain useful information learned from ADNI
     - **Adapt to AD-DECODE-specific inputs**, such as disease-related variation, blood pressure effects, and gene expression patterns
     - Improve age prediction on a broader age range and population (AD-DECODE includes young to old subjects, and at-risk individuals)



###  What the model learns from ADNI

By loading pretrained GATv2 weights from ADNI, the model starts with a strong initialization that:

- Encodes structural aging patterns 
- Captures useful representations of connectome topology 
- Reduces the risk of overfitting on small target datasets (like AD-DECODE)

The fine-tuning phase allows the model to **refine these representations**, adjusting them based on:

- **New node features** (MD)
- **Additional global context** (blood pressure, PCA genes)
- **Wider and younger age ranges**







On both:

###  Preprocessing:

- Connectomes:
  - Thresholded at the top 5% of connections
  - Log(x + 1) transformation applied
- Node features:
  - Z-score normalization (node-wise)
- Global features:
  - Encoded and concatenated with graph metrics

###  Architecture:

- 4-layer GATv2 with residual connections  
- Batch normalization and dropout  
- Global feature MLP with fusion  
- Final MLP head for brain age regression

###  Training Strategy:

- 7-fold stratified cross-validation by age  
- 10 repetitions per fold  
- Early stopping (patience = 40)  
- Loss: SmoothL1Loss  
- Optimizer: AdamW  
- Scheduler: StepLR  

