## 4.1 Predict on all normalizing with healthy
- Uses the pretrained model on healthy subjects and applies it to all subjects
- Predicts brain age for each subject (all risks)
- Computes BAG and cBAG
- Vizualizes results
- Saves CSV 

## 4.1a_clipping 
clips predicted brain ages at 120, 
recalculates the Brain Age Gap (BAG) and corrected BAG (cBAG), 
and saves the updated results.

## 4.1b_BAG and VIOLIN PLOTS
- Loads brain age predictions with metadata.
- Computes and visualizes BAG and corrected BAG (cBAG) vs chronological age.
- Runs Kruskal–Wallis and Mann–Whitney U tests for BAG/cBAG across groups.
- Saves statistical results and significance markers.
- Generates violin plots of BAG and cBAG

## 4.1b_Non parametric
Runs Kruskal–Wallis tests to detect global group differences in BAG and cBAG.
Performs pairwise Mann–Whitney U tests between group combinations.
Annotates each test with significance stars (e.g., *, **).
Saves the statistical results to CSV files per metric (BAG, cBAG).

## 4.2 metrics  AUC acc prec  recall
Computes classification performance metrics (AUC, Accuracy, Precision, Recall) based on cBAG thresholds.

## 5.0 SHAP and beeswarms
Computes shap global values and Visualizes global feature importance using SHAP beeswarm plots.



## FOLDERS 5.2 shap edges dti and 5.3 shap edges  fmri 
**SHAP edges.py**  Computes SHAP values for graph edges (connectome connections).
**glass brain top 10 mean.py**  Displays the top 10 most important edges (by mean SHAP) on a glass brain plot 
**glass brain top 10 mean left right inter.py**  Represents top 10 SHAP edges into left, right, and interhemispheric categories in one glass brain plot.




## 6_ADRC Regression BAG cognition
cBAG against cognitive test scores.

## 7_ADRC Regression BAG volumes
cBAG against regional brain volumes to identify anatomical correlates of accelerated aging.

## 9 ClusterSubjects
 **UMAP for 2D embedding** and KMeans to cluster subjects 
 Assigns cluster labels and saves both the UMAP plot and CSV with cluster IDs.
 Merges cluster labels with metadata (Risk, Sex, APOE).
 Plots cluster-wise composition using stacked bar plots:
   SHAP cluster by Risk group
   SHAP cluster by APOE genotype
   SHAP cluster by Sex

## 10_ Dendogram
Loads SHAP values and selected metadata (Sex, Genotype, graph metrics, biomarkers).
Z-scores SHAP values and performs KMeans clustering (k=3) for subject grouping.
**Builds a hierarchical clustermap** (heatmap + dendrograms) of SHAP profiles:
Rows = subjects, colored by cluster
Columns = features, colored by feature group (demographic, graph metrics, etc.)
Generates two clustermaps:
One with cluster-based row annotations.
One without cluster coloring.


