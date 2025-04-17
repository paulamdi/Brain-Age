# ADDECODE

This folder contains scripts for working with the AD-DECODE dataset

## Scripts

### 1_Uploading ADDECODE Data.py

This script loads brain connectivity matrices (connectomes) from compressed files, processes the associated metadata, and filters out non-healthy subjects.

- Loads connectome data (`.csv`) from a `.zip` file, skipping not needed files (e.g., white matter).
- Extracts subject IDs from filenames and standardizes them.
- Loads clinical metadata from a `.csv` file and processes the `DWI` field to create unique subject identifiers.
- Matches connectome matrices with metadata based on subject IDs.
- Filters out rows with missing data and drops Alzheimer’s Disease (AD) and Mild Cognitive Impairment (MCI) subjects, keeping only healthy controls.
- Displays a sample connectome and visualizes it using a heatmap.

