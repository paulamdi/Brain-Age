#SHAP CL
#zscored shap 
#topk

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os



#SHAP CSV and ZSCORE

from scipy.stats import zscore

# === Load SHAP CSV ===

# Load SHAP vectors without age
df_shap = pd.read_csv("FULL_FINE_TUNING_shap_global_features_no_age.csv")

# Extract subject IDs
subject_ids = df_shap["Subject_ID"].values

# Extract SHAP values and apply z-score normalization (across subjects, axis=0)
shap_vectors_raw = df_shap.drop(columns=["Subject_ID"]).values
shap_vectors_zscored = zscore(shap_vectors_raw, axis=0)

# Optional: convert back to DataFrame if you want to save or inspect
df_zscored = pd.DataFrame(shap_vectors_zscored, columns=df_shap.columns[1:], index=subject_ids)
df_zscored.insert(0, "Subject_ID", subject_ids)

# Save if needed
df_zscored.to_csv("FULL_FINE_TUNING_shap_global_features_zscored.csv", index=False)




# SIMILARITY MATRIX

# === Compute Cosine Similarity Matrix ===
# Compute the cosine similarity between each pair of SHAP vectors
# This gives us a matrix where [i, j] = similarity between subject i and subject j


# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(shap_vectors_zscored)

# Optionally save
pd.DataFrame(similarity_matrix, index=subject_ids, columns=subject_ids).to_csv("FULL_FINE_TUNING_shap_similarity_matrix.csv")



# SIMILARITY HISTOGRAM TH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load similarity matrix ===
sim_df = pd.read_csv("FULL_FINE_TUNING_shap_similarity_matrix.csv", index_col=0)
sim_matrix = sim_df.values

# === Extract upper triangle values (excluding the diagonal) ===
triu_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]  # k=1 removes diagonal

# === Plot histogram ===
plt.figure(figsize=(8, 5))
plt.hist(triu_values, bins=50, color='lightblue', edgecolor='k')
plt.axvline(0.7, color='green', linestyle='--', label='Positive Threshold (0.8)')
plt.axvline(0.3, color='red', linestyle='--', label='Negative Threshold (0.2)')
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Subject Pairs")
plt.title("Distribution of SHAP-Based Similarities Between Subjects")
plt.legend()
plt.tight_layout()
plt.show()






#FUNCTION TO GENEATE PAIRS

# For each subject (anchor), we select:

    
# We sample up to n_pairs of each to form triplets: (anchor, positive, negative)

def generate_pairs_topk(sim_matrix, subject_ids, k_pos=5, k_neg=5):
    """
    For each subject (anchor), select:
      - top k_pos most similar subjects (positive)
      - bottom k_neg least similar subjects (negative)
    to create triplets of the form (anchor, positive, negative)
    """
    n_subjects = len(subject_ids)
    pairs = []

    for i in range(n_subjects):
        anchor_id = subject_ids[i]
        similarities = sim_matrix[i].copy()

        # Remove self-comparison
        similarities[i] = -np.inf

        # Get top-k most similar (positive) indices
        pos_indices = similarities.argsort()[::-1][:k_pos]

        # Get bottom-k least similar (negative) indices
        neg_indices = similarities.argsort()[:k_neg]

        # Form triplets
        for pos in pos_indices:
            for neg in neg_indices:
                pairs.append((anchor_id, subject_ids[pos], subject_ids[neg]))

    return pairs




# TRIPLETS

# Create the list of triplets (anchor, positive, negative)
# These triplets will later be used for training with contrastive loss

# Generate SHAP-based triplets
# Create triplets using top-k strategy

triplets_topk = generate_pairs_topk(similarity_matrix, subject_ids, k_pos=5, k_neg=5)

# Save to CSV
triplet_df_topk = pd.DataFrame(triplets_topk, columns=["anchor", "positive", "negative"])
triplet_df_topk.to_csv("FULL_FINE_TUNING_shap_triplets_topk.csv", index=False)





#HOW MANY TIMES EACH SUBJECT APPEARS

from collections import Counter


anchor_counts = Counter(triplet_df_topk["anchor"])
positive_counts = Counter(triplet_df_topk["positive"])
negative_counts = Counter(triplet_df_topk["negative"])

all_ids = set(triplet_df_topk["anchor"]) | set(triplet_df_topk["positive"]) | set(triplet_df_topk["negative"])

summary_topk = pd.DataFrame({
    "Subject_ID": list(all_ids),
    "Anchor": [anchor_counts.get(i, 0) for i in all_ids],
    "Positive": [positive_counts.get(i, 0) for i in all_ids],
    "Negative": [negative_counts.get(i, 0) for i in all_ids],
})

summary_topk.to_csv("FULL_FINE_TUNING_triplet_topk_role_summary.csv", index=False)
