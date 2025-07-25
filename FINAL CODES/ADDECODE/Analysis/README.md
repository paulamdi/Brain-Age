AD-DECODE Brain Age Prediction Pipeline

# 0_CountingSubjectsADDECODE.py
**Counts subjects** with available connectopmes, gloal features, node features, PCA genes. 
Counts by sex, risk groups, APOE genotype.




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

