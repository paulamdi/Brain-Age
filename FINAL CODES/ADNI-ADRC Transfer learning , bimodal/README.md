

## 1_ADNI Pretraining (Bimodal GATv2):
Pretrains a Dual-GATv2 model on healthy ADNI subjects using DTI and fMRI connectomes, node features (FA, Volume), and global metadata.
The model learns to predict brain age through early fusion of graph and demographic features.
Saves model trained on all healthy

## 2_ADRC Fine-Tuning (ADNI → ADRC Transfer):
Full Fine-tunes the pretrained ADNI bimodal model on non-demented ADRC subjects using matched DTI/fMRI data.
It evaluates brain age prediction via repeated stratified 7-fold cross-validation with performance metrics.

Fix metadata!!
