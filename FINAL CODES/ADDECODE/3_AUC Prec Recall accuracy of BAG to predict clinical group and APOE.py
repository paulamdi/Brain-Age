#ADDECODE

# Evaluating BAG / cBAG as a biomarker to distinguish clinical risk groups and genotype
# ROC curve, AUC, Accuracy, Recall, Precision, F1-score


import pandas as pd

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import os

import os

# === Define main directory where all figures will be saved ===
BASE_DIR = "metrics"
os.makedirs(BASE_DIR, exist_ok=True)

def make_dirs(*path_parts):
    path = os.path.join(BASE_DIR, *path_parts)
    os.makedirs(path, exist_ok=True)
    return path

def save_results_txt(results_dict, save_dir, filename="results.txt"):
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value:.3f}\n")





# === 1. Load CSV with predictions and metadata ===
df = pd.read_csv("brain_age_predictions_with_metadata.csv")





#APOE

#-> BAG
# === Define binary target: 1 = APOE E4+, 0 = E4−
df['Group'] = (df['APOE'] == 'E4+').astype(int)
y_true = df['Group'].values
y_score = df['BAG'].values

# === ROC & threshold
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("APOE using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save everything
save_dir = make_dirs("APOE", "BAG")
results_dict = {
    "AUC": auc_value, "Threshold": best_threshold,
    "Accuracy": acc, "Recall": rec,
    "Precision": prec, "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve for APOE E4+ Prediction using BAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["E4−", "E4+"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: APOE E4+ Prediction using BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# Violin Plot
df["APOE_Label"] = df["Group"].map({0: "E4−", 1: "E4+"})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x="APOE_Label", y="BAG", palette="Set2", inner="box", order=["E4−", "E4+"])
plt.title("BAG Distribution by APOE Status (AD-DECODE)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()

print()



#APOE

#-> cBAG
# === Define binary target: 1 = APOE E4+, 0 = E4−
df['Group'] = (df['APOE'] == 'E4+').astype(int)
y_true = df['Group'].values
y_score = df['cBAG'].values

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Best threshold by Youden's J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print to console
print("APOE using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("APOE", "cBAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve for APOE E4+ Prediction using cBAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["E4−", "E4+"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: APOE E4+ Prediction using cBAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot
df["APOE_Label"] = df["Group"].map({0: "E4−", 1: "E4+"})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x="APOE_Label", y="cBAG", palette="Set2", inner="box", order=["E4−", "E4+"])
plt.title("cBAG Distribution by APOE Status (AD-DECODE)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()




print()





# RISK 

#AD vs No risk
# -> BAG

# === Filter data: AD vs NoRisk
df_risk = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()
df_risk['Group'] = (df_risk['Risk'] == 'AD').astype(int)
y_true = df_risk['Group'].values
y_score = df_risk['BAG'].values

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print results
print("RISK: AD vs NoRisk using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("RISK_AD_vs_NoRisk", "BAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: AD vs NoRisk Prediction using BAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# CHECK WHY METRICS ARE SO GREAT
# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Risk", "AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: RISK Prediction using BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_risk, x='Risk', y='BAG', palette="Set2")
plt.title("BAG Distribution: AD vs NoRisk")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()






#using cbag

#AD vs No risk 

# === Filter data: AD vs NoRisk
df_risk = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()
df_risk['Group'] = (df_risk['Risk'] == 'AD').astype(int)
y_true = df_risk['Group'].values
y_score = df_risk['cBAG'].values

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Compute performance metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print results
print("RISK: AD vs NoRisk using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("RISK_AD_vs_NoRisk", "cBAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: AD vs NoRisk Prediction using cBAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Risk", "AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: RISK Prediction using cBAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_risk, x='Risk', y='cBAG', palette="Set2")
plt.title("cBAG Distribution: AD vs NoRisk")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()





# NoRisk + Familial vs MCI + AD

#BAG

# === Filter data: NoRisk + Familial vs MCI + AD
df_risk = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_risk['Group'] = df_risk['Risk'].isin(['MCI', 'AD']).astype(int)
y_true = df_risk['Group'].values
y_score = df_risk['BAG'].values

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("RISK: NoRisk+Familial vs MCI+AD using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("RISK_FamNoRisk_vs_MCIAD", "BAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: NoRisk+Familial vs MCI+AD using BAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Impairment", "MCI/AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: NoRisk+Familial vs MCI+AD using BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_plot['Group'] = df_plot['Risk'].apply(lambda x: 'NoRisk+Familial' if x in ['NoRisk', 'Familial'] else 'MCI+AD')

plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Group', y='BAG', palette="Set2")
plt.title("BAG Distribution: NoRisk+Familial vs MCI+AD")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()



#cBAG



# === Filter data: NoRisk + Familial vs MCI + AD
df_risk = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_risk['Group'] = df_risk['Risk'].isin(['MCI', 'AD']).astype(int)
y_true = df_risk['Group'].values
y_score = df_risk['cBAG'].values

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("RISK: NoRisk+Familial vs MCI+AD using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("RISK_FamNoRisk_vs_MCIAD", "cBAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: NoRisk+Familial vs MCI+AD using cBAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Impairment", "MCI/AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: NoRisk+Familial vs MCI+AD using cBAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_plot['Group'] = df_plot['Risk'].apply(lambda x: 'NoRisk+Familial' if x in ['NoRisk', 'Familial'] else 'MCI+AD')

plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Group', y='cBAG', palette="Set2")
plt.title("cBAG Distribution: NoRisk+Familial vs MCI+AD")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin.png"))
plt.close()


# SEX

# BAG

# === Define binary target: 1 = Female, 0 = Male
df['Group'] = (df['sex'] == 'F').astype(int)
y_true = df['Group'].values
y_score = df['BAG'].values

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("SEX prediction using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save results
save_dir = make_dirs("SEX", "BAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: Sex Prediction using BAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Sex Prediction using BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot — All subjects
df["Sex_Label"] = df["sex"].map({'F': 'Female', 'M': 'Male'})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='Sex_Label', y='BAG', palette="pastel", inner="box")
plt.title("BAG by Sex — All Subjects")
plt.xlabel("Sex")
plt.ylabel("BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin_AllSubjects.png"))
plt.close()

# === Violin plot — Only healthy (NoRisk + Familial)
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial'])].copy()
df_plot['Sex_Label'] = df_plot['sex'].map({'F': 'Female', 'M': 'Male'})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Sex_Label', y='BAG', palette="pastel", inner="box")
plt.title("BAG by Sex — NoRisk + Familial Only")
plt.xlabel("Sex")
plt.ylabel("BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin_HealthyOnly.png"))
plt.close()


#cBAG

# === SEX: Female vs Male (using cBAG) ===

# === 1 = Female, 0 = Male
df['Group'] = (df['sex'] == 'F').astype(int)
y_true = df['Group'].values
y_score = df['cBAG'].values

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# === Metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("SEX prediction using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Save
save_dir = make_dirs("SEX", "cBAG")
results_dict = {
    "AUC": auc_value,
    "Threshold": best_threshold,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1-score": f1
}
save_results_txt(results_dict, save_dir)

# === ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve: Sex Prediction using cBAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ROC.png"))
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Sex Prediction using cBAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
plt.close()

# === Violin plot — All subjects
df["Sex_Label"] = df["sex"].map({'F': 'Female', 'M': 'Male'})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='Sex_Label', y='cBAG', palette="pastel", inner="box")
plt.title("Corrected BAG (cBAG) by Sex — All Subjects")
plt.xlabel("Sex")
plt.ylabel("Corrected BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin_AllSubjects.png"))
plt.close()

# === Violin plot — Only healthy (NoRisk + Familial)
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial'])].copy()
df_plot['Sex_Label'] = df_plot['sex'].map({'F': 'Female', 'M': 'Male'})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Sex_Label', y='cBAG', palette="pastel", inner="box")
plt.title("Corrected BAG by Sex (NoRisk + Familial only)")
plt.xlabel("Sex")
plt.ylabel("Corrected BAG")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Violin_HealthyOnly.png"))
plt.close()









#PRIORITIZING RECALL

# --------------------------------------------
# APOE — Prioridad: Recall ≥ 0.60 (BAG & cBAG)
# --------------------------------------------

recall_target = 0.60
df["Group"] = (df["APOE"] == "E4+").astype(int)
y_true = df["Group"].values

for score_type in ["BAG", "cBAG"]:
    # ------------- ROC & threshold -------------
    y_score = df[score_type].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    # Umbral que asegura recall ≥ recall_target
    idx_candidates = np.where(tpr >= recall_target)[0]
    if len(idx_candidates):
        best_idx = idx_candidates[0]
        best_threshold = thresholds[best_idx]
    else:                               # fallback a Youden
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype(int)

    # ------------- Métricas -------------
    acc  = accuracy_score (y_true, y_pred)
    rec  = recall_score   (y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1   = f1_score       (y_true, y_pred)

    print(f"\n=== APOE • {score_type} (Recall ≥ {recall_target}) ===")
    print(f"AUC: {auc_value:.3f}")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # ------------- Guardar resultados -------------
    save_dir = make_dirs("APOE_recall60", score_type)
    save_results_txt(
        {
            "AUC": auc_value,
            "Threshold": best_threshold,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1-score": f1
        },
        save_dir
    )

    # ------------- ROC -----------------
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red',
                label=f"Thr = {best_threshold:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC • APOE E4+ • {score_type} (Recall ≥ {recall_target})")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ROC.png"))
    plt.close()

    # ------------- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["E4−", "E4+"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion • {score_type} (Recall ≥ {recall_target})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
    plt.close()

    # ------------- Violin plot ----------
    df["APOE_Label"] = df["Group"].map({0: "E4−", 1: "E4+"})
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="APOE_Label", y=score_type,
                   palette="Set2", inner="box",
                   order=["E4−", "E4+"])
    plt.title(f"{score_type} Distribution by APOE (Recall ≥ {recall_target})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Violin.png"))
    plt.close()


# --------------------------------------------
# RISK — Recall ≥ 0.60 • NoRisk+Familial vs MCI+AD
# --------------------------------------------

# ── Prepare data ────────────────────────────
df_risk = df[df["Risk"].isin(["NoRisk", "Familial", "MCI", "AD"])].copy()
df_risk["Group"] = df_risk["Risk"].isin(["MCI", "AD"]).astype(int)   # 1 = MCI/AD, 0 = NoRisk/Familial
y_true = df_risk["Group"].values

recall_target = 0.60

# ── Evaluate BAG and cBAG ───────────────────
for score_type in ["BAG", "cBAG"]:
    y_score = df_risk[score_type].values

    # ----- ROC & threshold (recall priority) -----
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    idx_candidates = np.where(tpr >= recall_target)[0]
    if len(idx_candidates):
        best_idx = idx_candidates[0]
        best_threshold = thresholds[best_idx]
    else:                       # fallback → maximise Youden’s J
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype(int)

    # ----- Metrics -----
    acc  = accuracy_score (y_true, y_pred)
    rec  = recall_score   (y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1   = f1_score       (y_true, y_pred)

    print(f"\n=== RISK • {score_type} (Recall ≥ {recall_target}) ===")
    print(f"AUC:        {auc_value:.3f}")
    print(f"Threshold:  {best_threshold:.3f}")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Recall:     {rec:.3f}")
    print(f"Precision:  {prec:.3f}")
    print(f"F1-score:   {f1:.3f}")

    # ----- Save metrics -----
    save_dir = make_dirs("RISK_recall60", score_type)
    save_results_txt(
        {
            "AUC": auc_value,
            "Threshold": best_threshold,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1-score": f1
        },
        save_dir
    )

    # ----- ROC curve -----
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red',
                label=f"Thr = {best_threshold:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC • RISK NoRisk+Familial vs MCI+AD • {score_type} (Recall ≥ {recall_target})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ROC.png"))
    plt.close()

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Impairment", "MCI/AD"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion • {score_type} (Recall ≥ {recall_target})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ConfusionMatrix.png"))
    plt.close()

    # ----- Violin plot -----
    df_plot = df_risk.copy()
    df_plot["Group_Label"] = df_plot["Risk"].apply(
        lambda x: "NoRisk+Familial" if x in ["NoRisk", "Familial"] else "MCI+AD"
    )
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df_plot, x="Group_Label", y=score_type,
                   palette="Set2", inner="box",
                   order=["NoRisk+Familial", "MCI+AD"])
    plt.title(f"{score_type} Distribution (Recall ≥ {recall_target})")
    plt.xlabel("Risk Group"); plt.ylabel(score_type)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Violin.png"))
    plt.close()
