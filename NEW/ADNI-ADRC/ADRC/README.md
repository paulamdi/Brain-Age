## Brain Age Prediction Models using ADRC Connectomes

This repository contains three different Graph Neural Network (GNN) pipelines for predicting brain age from connectomes using the ADRC dataset. All models are based on the GATv2 architecture and incorporate both node-level neuroimaging features and subject-level metadata.

---

### `1_dti_only/` — DTI-Only Brain Age Prediction

This pipeline trains a GATv2 model on structural DTI connectomes only.

- Node features: Fractional Anisotropy (FA) and Volume per brain region.
- Global features: Sex, APOE genotype, and DTI graph metrics (clustering coefficient, path length, global/local efficiency).
- Graph: 84×84 thresholded and log-transformed DTI adjacency matrix.
- Model: 4-layer GATv2 with residual connections and global metadata fusion.
- Evaluation: Repeated stratified 7-fold cross-validation and final model training on all healthy subjects.

Outputs: Brain age predictions, MAE, R², RMSE, learning curves, and pretrained final model.

---

### `2_dti_fmri_separate/` — DTI and fMRI as Independent Pipelines

This script runs two separate models, one for DTI and one for fMRI, to compare performance across modalities.

- DTI pipeline:
  - Same setup as `1_dti_only`, using only subjects with both DTI and fMRI data.
- fMRI pipeline:
  - Uses correlation-based fMRI connectomes.
  - Same node features (FA, Volume) and global features (sex, genotype, fMRI graph metrics).
- Modality fusion: No fusion — DTI and fMRI are trained and evaluated independently.

Outputs: Separate performance metrics (MAE, R², RMSE) for DTI and fMRI using matched subjects.

---

### `3_dti_fmri_fused/` — Early Fusion of DTI and fMRI

This pipeline implements a multimodal GATv2 model with early fusion of DTI and fMRI connectomes.

- Node features: Shared node features across modalities, including Fractional Anisotropy (FA) and Volume per brain region.
- Connectomes: Two 84×84 adjacency matrices (DTI and fMRI), both using the same node features.
- Global features: Sex, APOE genotype, and graph metrics from both DTI and fMRI.
- Fusion strategy: The two connectomes are processed in parallel by two GATv2 backbones (one for DTI, one for fMRI) that share node embeddings. Their outputs are concatenated with global metadata before the regression head.
- Model: Dual-branch GATv2 with shared node embeddings and a unified prediction head.
- Evaluation: Repeated 7-fold cross-validation on subjects with both modalities.

Outputs: Multimodal brain age predictions and evaluation of joint DTI + fMRI modeling.

