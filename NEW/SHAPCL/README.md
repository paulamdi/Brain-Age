###  `code_1 addecode_train_model_on_all_healthy.py` — Graph Construction and GATv2 Training on Healthy Subjects

This script builds graph data objects from AD-DECODE structural connectomes, integrating:

- **Multimodal node features**: FA, MD, Volume, and clustering coefficient.
- **Global metadata**: sex, genotype, systolic/diastolic blood pressure, graph-level metrics, and top 10 PCA gene components.

Key processing steps include percentile thresholding, log(x+1) transformation, and z-score normalization. The model architecture is a 4-layer GATv2 with residual connections, batch normalization, and multi-head attention. It is trained using **7-fold cross-validation with 10 repetitions**, and evaluated using **MAE, R², and learning curves**. 

A final model trained on all healthy subjects is saved for inference on MCI/AD cases.

### `code_2 addecode_infer_shap.py` — Inference and SHAP Analysis on AD-DECODE

- Applies a pretrained GATv2 model (trained on healthy controls) to all AD-DECODE subjects, including MCI and AD cases.
- Ensures consistent preprocessing by normalizing all node and global features using statistics computed only from healthy subjects.
- Constructs graph data objects with FA, MD, Volume, and clustering coefficient as node features, and demographic, graph metrics, and PCA components as global features.
- Uses SHAP (DeepExplainer) to compute global feature importance by wrapping the model’s metadata encoder with `GlobalOnlyModel`.
- Saves per-subject SHAP values to CSV (`shap_global_features_all_subjects.csv`) and a version excluding age (`shap_global_features_no_age.csv`) for contrastive learning.

  ### `code_3 shap_contrastive_triplets.py` — SHAP-Based Contrastive Triplet Generation

- Loads global SHAP values per subject (from previous model inference), applies z-score normalization across subjects, and saves the normalized matrix.
- Computes a **cosine similarity matrix** between all z-scored SHAP vectors, representing subject-level similarity in feature importance.
- Visualizes the distribution of pairwise SHAP-based similarities using a histogram, and identifies cutoffs for potential thresholds (e.g., >0.8 for positive pairs, <0.2 for negatives).
- Generates triplet samples `(anchor, positive, negative)` using a top-k strategy: selects the most similar and least similar subjects for each anchor based on cosine similarity.
- Saves both the generated triplets and a summary of how frequently each subject appears as anchor, positive, or negative for downstream use in contrastive learning.

### `code_4 shap_contrastive_learning.py` — SHAP Contrastive Embedding with Triplet NT-Xent Loss

- Loads z-scored global SHAP vectors and corresponding triplets `(anchor, positive, negative)` generated from cosine similarity.
- Implements a custom `TripletDataset` to construct batches for training using PyTorch’s `DataLoader`.
- Defines a **projection head** (MLP) that maps SHAP vectors into a lower-dimensional embedding space and normalizes outputs (L2).
- Implements a **triplet-based NT-Xent loss** that encourages anchors to be closer to their positives than to their negatives using cosine similarity and cross-entropy.
- Trains the embedding model for 100 epochs, saves the model weights (`shap_projection_head_trained.pt`), and outputs final embeddings for all subjects to `shap_embeddings.csv` for downstream clustering or classification.


### `code_5 model for brain age with SHAP embed.py` — Integrating SHAP-CL Embeddings into Brain Age Prediction

- Loads the SHAP-based embeddings (`shap_embeddings.csv`) and merges them into the existing AD-DECODE dataset as additional global features.
- Appends the 32-dimensional SHAP embeddings to each subject's feature vector, alongside demographics, graph metrics, and PCA genes.
- Uses the **original GATv2 brain age model** trained on AD-DECODE and fine-tunes it with the augmented feature set to assess the utility of SHAP embeddings.
- Performs training and evaluation (e.g., with 7-fold CV) to compare performance (MAE, R²) with and without SHAP-CL features.
- Provides insight into whether learned SHAP embeddings capture meaningful subject-level variation that improves brain age prediction.
