- Using model trained on all healthy 

-BAG and corrected BAG (cBAG) computation and violin plots comparing BAG/cBAG across clinical risk groups, APOE...

-Performs non-parametric statistical tests (Kruskal-Wallis, Mann-Whitney) to assess group differences in BAG.

-Computes classification performance metrics (AUC, Accuracy, Precision, Recall) based on cBAG thresholds.

-Visualizes global feature importance using SHAP beeswarm plots.

-Computes SHAP values for edge attributes (connectome) using a GNN gradient-based method

-Plot mean SHAP edge importance matrices

-Compares edge-level SHAP values between different age groups

-Creates glass brain visualizations of the most important edges (top-10) across the dataset using Nilearn.

-Regresses cBAG against cognitive test scores.

-Regresses cBAG against regional brain volumes to identify anatomical correlates of accelerated aging.

-Performs unsupervised clustering (e.g., KMeans, UMAP) based on SHAP values to reveal distinct subject profiles.

-Constructs dendrograms based on SHAP vectors and overlays metadata

