
import os  # For handling file paths and directories
import pandas as pd  # For working with tabular data using DataFrames
import matplotlib.pyplot as plt  # For generating plots
import seaborn as sns  # For enhanced visualizations of heatmaps
import zipfile  # For reading compressed files without extracting them
import re  # For extracting numerical IDs using regular expressions

import torch
import random
import numpy as np

import os
import pandas as pd
import numpy as np

import networkx as nx  # For graph-level metrics


# === Load predictions (only subject_id, age, predicted_age, BAG/cBAG) ===
df_preds = pd.read_csv("/home/bas/Desktop/Columns/column_model_predictions_all_subjects_with_cBAG.csv")

# Pad subject IDs to 5 digits if needed
df_preds["subject_id"] = df_preds["subject_id"].astype(str).str.zfill(5)

# === Load AD-DECODE metadata ===
df_meta = pd.read_excel("/home/bas/Desktop/MyData/AD_DECODE/AD_DECODE_data4.xlsx")

# Fix format of metadata IDs
df_meta["MRI_Exam_fixed"] = df_meta["MRI_Exam"].fillna(0).astype(int).astype(str).str.zfill(5)

# === Merge metadata with predictions ===
df = pd.merge(df_preds, df_meta, left_on="subject_id", right_on="MRI_Exam_fixed", how="left")


from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Create binary label for APOE4+
df["Group"] = (df["APOE"] == "E4+").astype(int)
y_true = df["Group"].values
y_score = df["cBAG"].values  # You can switch to df["BAG"].values if needed

# === Step 2: Fixed threshold = 0
y_pred_fixed = (y_score > 0).astype(int)

# === Step 3: Optimal threshold using Youden’s J
fpr, tpr, thresholds = roc_curve(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred_opt = (y_score >= best_threshold).astype(int)

# === Step 4: Metrics
print("=== Fixed Threshold (cBAG > 0) ===")
print(f"AUC:       {roc_auc_score(y_true, y_score):.3f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_fixed):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred_fixed):.3f}")
print(f"Precision: {precision_score(y_true, y_pred_fixed):.3f}")
print(f"F1-score:  {f1_score(y_true, y_pred_fixed):.3f}")

print("\n=== Optimal Threshold (Youden's J) ===")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred_opt):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred_opt):.3f}")
print(f"Precision: {precision_score(y_true, y_pred_opt):.3f}")
print(f"F1-score:  {f1_score(y_true, y_pred_opt):.3f}")


# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_true, y_score):.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Optimal Thresh = {best_threshold:.2f}")
plt.axvline(fpr[np.abs(thresholds - 0).argmin()], color='blue', linestyle='--', label="Thresh = 0")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: APOE4+ Prediction using cBAG")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix (optimal threshold)
cm = confusion_matrix(y_true, y_pred_opt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["E4−", "E4+"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Optimal Threshold)")
plt.tight_layout()
plt.show()
