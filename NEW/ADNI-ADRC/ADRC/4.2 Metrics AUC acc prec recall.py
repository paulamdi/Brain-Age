


# Evaluating BAG / cBAG as a biomarker to distinguish clinical risk groups and genotype
# ROC curve, AUC, Accuracy, Recall, Precision, F1-score


import pandas as pd

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer




# === 1. Load CSV with predictions and metadata ===
df = pd.read_csv("/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/brainage_predictions_adrc_all_with_metadata.csv")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)

# === Output directory
output_dir = "metrics"
os.makedirs(output_dir, exist_ok=True)



# APOE



# === APOE conversion
def apoe_status(apoe_str):
    if isinstance(apoe_str, str):
        return 'E4+' if '4' in apoe_str else 'E4-'
    return np.nan

df['APOE_status'] = df['APOE'].apply(apoe_status)
df = df[df['APOE_status'].isin(['E4+', 'E4-'])].copy()
df['Group'] = (df['APOE_status'] == 'E4+').astype(int)
y_true = df['Group'].values

# === Run for both BAG and cBAG
for predictor in ['BAG', 'cBAG']:
    y_score = df[predictor].values

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)

    # Metrics
    metrics_df = pd.DataFrame({
        'AUC': [auc_value],
        'Best_Threshold': [best_threshold],
        'Accuracy': [accuracy_score(y_true, y_pred)],
        'Recall': [recall_score(y_true, y_pred)],
        'Precision': [precision_score(y_true, y_pred)],
        'F1': [f1_score(y_true, y_pred)],
        'N_subjects': [len(df)]
    })

    # Save CSV
    csv_path = os.path.join(output_dir, f"metrics_apoe_{predictor.lower()}.csv")
    metrics_df.to_csv(csv_path, index=False)

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.title(f"ROC Curve for APOE E4+ Prediction using {predictor} (ADRC)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f"roc_apoe_{predictor.lower()}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

print(" All figures and metrics saved to 'metrics/' folder.")



#RISK 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# === Create output directory
output_dir = "metrics"
os.makedirs(output_dir, exist_ok=True)

# === Define new risk group: NoRisk vs Sick (Demented or IMPNOMCI)
def assign_risk(row):
    if row["NORMCOG"] == 1:
        return "NoRisk"
    elif row["DEMENTED"] == 1 or row["IMPNOMCI"] == 1:
        return "Sick"
    else:
        return "Unknown"

df["RiskGroup"] = df.apply(assign_risk, axis=1)

# === Filter to Sick vs NoRisk
df_risk = df[df["RiskGroup"].isin(["Sick", "NoRisk"])].copy()
df_risk["Group"] = (df_risk["RiskGroup"] == "Sick").astype(int)
y_true = df_risk["Group"].values

# === Loop over BAG and cBAG
for predictor in ["BAG", "cBAG"]:
    y_score = df_risk[predictor].values

    # ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)

    # Save metrics
    metrics_df = pd.DataFrame({
        'AUC': [auc_value],
        'Best_Threshold': [best_threshold],
        'Accuracy': [accuracy_score(y_true, y_pred)],
        'Recall': [recall_score(y_true, y_pred)],
        'Precision': [precision_score(y_true, y_pred)],
        'F1': [f1_score(y_true, y_pred)],
        'N_subjects': [len(df_risk)]
    })
    csv_path = os.path.join(output_dir, f"metrics_risk_sick_vs_norisk_{predictor.lower()}.csv")
    metrics_df.to_csv(csv_path, index=False)

    # ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.title(f"ROC Curve: Sick vs NoRisk using {predictor}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_risk_sick_vs_norisk_{predictor.lower()}.png"), dpi=300)
    plt.close()

    # Violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df_risk, x='RiskGroup', y=predictor, palette="Set2")
    plt.title(f"{predictor} Distribution: Sick vs NoRisk")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_risk_sick_vs_norisk_{predictor.lower()}.png"), dpi=300)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NoRisk", "Sick"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: Sick vs NoRisk ({predictor})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_risk_sick_vs_norisk_{predictor.lower()}.png"), dpi=300)
    plt.close()

print(" Finished: Sick (Demented + IMPNOMCI) vs NoRisk using BAG and cBAG → saved to 'metrics/'")


#SEX

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# === Output directory
output_dir = "metrics"
os.makedirs(output_dir, exist_ok=True)

# === Convert SUBJECT_SEX to sex labels
df['sex'] = df['SUBJECT_SEX'].map({1: 'F', 2: 'M'})  # 1 = Female, 2 = Male

# === Binary classification: 1 = Female, 0 = Male
df['Group'] = (df['sex'] == 'F').astype(int)
y_true = df['Group'].values

# === Loop over both BAG and cBAG
for predictor in ['BAG', 'cBAG']:
    y_score = df[predictor].values

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)

    # Metrics
    metrics_df = pd.DataFrame({
        'AUC': [auc_value],
        'Best_Threshold': [best_threshold],
        'Accuracy': [accuracy_score(y_true, y_pred)],
        'Recall': [recall_score(y_true, y_pred)],
        'Precision': [precision_score(y_true, y_pred)],
        'F1': [f1_score(y_true, y_pred)],
        'N_subjects': [len(df)]
    })
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_sex_{predictor.lower()}.csv"), index=False)

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.title(f"ROC Curve: Sex Prediction using {predictor}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_sex_{predictor.lower()}.png"), dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: Sex Prediction using {predictor}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_sex_{predictor.lower()}.png"), dpi=300)
    plt.close()

    # Violin (all)
    df['Sex_Label'] = df['sex'].map({'F': 'Female', 'M': 'Male'})
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x='Sex_Label', y=predictor, palette="pastel", inner="box")
    plt.title(f"{predictor} Distribution by Sex (All Subjects)")
    plt.xlabel("Sex")
    plt.ylabel(predictor)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"violin_sex_{predictor.lower()}.png"), dpi=300)
    plt.close()

   
print(" SEX analysis completed for BAG and cBAG — saved in 'metrics/'")





#SUMMARY METRICS
import os
import pandas as pd

# === Folder where all metrics CSVs are stored
metrics_dir = "metrics"

# === List to store all metric DataFrames
all_metrics = []

# === Loop through all CSV files starting with 'metrics_'
for fname in os.listdir(metrics_dir):
    if fname.startswith("metrics_") and fname.endswith(".csv"):
        path = os.path.join(metrics_dir, fname)
        df = pd.read_csv(path)

        # Extract base name without extension
        base = fname.replace("metrics_", "").replace(".csv", "")
        parts = base.split("_")
        target = parts[0]        # e.g., 'apoe', 'risk', 'sex'
        predictor = parts[-1]    # e.g., 'bag', 'cbag'

        # Add identifier columns
        df["Target"] = target
        df["Predictor"] = predictor.upper()  # Make it uppercase for consistency

        # Append to the list
        all_metrics.append(df)

# === Concatenate all metric DataFrames
summary_df = pd.concat(all_metrics, ignore_index=True)

# === Optional: reorder columns for clarity
cols = ['Target', 'Predictor', 'AUC', 'Accuracy', 'Recall', 'Precision', 'F1', 'Best_Threshold', 'N_subjects']
summary_df = summary_df[cols]

# === Save the summary to a single CSV
summary_path = os.path.join(metrics_dir, "summary_all_metrics.csv")
summary_df.to_csv(summary_path, index=False)

print(f" Saved summary table to: {summary_path}")

