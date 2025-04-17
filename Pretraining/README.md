# Pretraining

This folder contains scripts related to data preparation and correlation analysis used for pretraining models in the Brain-Age project. The code works with the ADNI and AD-DECODE datasets.

## Scripts

### 1_Uploading ADNI Data.py

This script loads and prepares brain connectivity matrices (connectomes) and clinical metadata from the ADNI dataset for use in pretraining.

- Loads connectome data (`.csv` matrices) from subject-specific folders.
- Loads and filters clinical metadata from an Excel file.
- Generates standardized connectome keys by combining subject ID and visit code (e.g., `R4288_y0`).
- Matches each connectome to its corresponding subject metadata.
- Filters to include only healthy control (CN) subjects.
- Selects a random subject and visualizes their connectome matrix as a heatmap using Seaborn.

### 2_PearsonCorrelation.py

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


