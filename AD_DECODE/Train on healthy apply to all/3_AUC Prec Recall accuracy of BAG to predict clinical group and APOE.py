# Evaluating BAG / cBAG as a biomarker to distinguish clinical risk groups and genotype


import pandas as pd

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# === 1. Load CSV with predictions and metadata ===
df = pd.read_csv("/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/brain_age_predictions_with_metadata.csv")





#APOE

#-> BAG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)

# === Define binary target: 1 = APOE E4+, 0 = E4−
df['Group'] = (df['APOE'] == 'E4+').astype(int)
y_true = df['Group'].values

# === Use BAG or cBAG as predictor
y_score = df['BAG'].values  # Or use df['cBAG'].values for corrected BAG

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Apply best threshold to binarize predictions
y_pred = (y_score >= best_threshold).astype(int)

# === Compute evaluation metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print metrics
print(" APOE using Bag")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.3f})")
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f"Best Threshold = {best_threshold:.2f}")
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
plt.title("ROC Curve for APOE E4+ Prediction using BAG")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print()



#APOE

#-> cBAG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)

# === Define binary target: 1 = APOE E4+, 0 = E4−
df['Group'] = (df['APOE'] == 'E4+').astype(int)
y_true = df['Group'].values

# === Use BAG or cBAG as predictor
y_score = df['cBAG'].values  # Or use df['cBAG'].values for corrected BAG

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Apply best threshold to binarize predictions
y_pred = (y_score >= best_threshold).astype(int)

# === Compute evaluation metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print metrics
print(" APOE using CBag")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Plot ROC curve
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
plt.show()

print()


# RISK 

#AD vs No risk


# === Filter data: AD vs NoRisk
df_risk = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()
df_risk['Group'] = (df_risk['Risk'] == 'AD').astype(int)

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np

# === Define true labels and BAG scores
y_true = df_risk['Group'].values
y_score = df_risk['BAG'].values  # You can switch to df_risk['cBAG'] if desired

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Apply threshold to classify
y_pred = (y_score >= best_threshold).astype(int)

# === Compute performance metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print results
print(" RISK: AD vs NoRisk using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Plot ROC curve
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
plt.show()

print()

#CHECK WITH METRICS ARE SO GREAT

# Filter AD and No risk
df_subset = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()

# Print
print(df_subset[['MRI_Exam_fixed', 'Risk', 'Age', 'Predicted_Age', 'BAG']].sort_values('BAG', ascending=False))

print()


import seaborn as sns
import matplotlib.pyplot as plt

# Violin plot BAG vs group
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_subset, x='Risk', y='BAG', palette="Set2")
plt.title("BAG Distribution: AD vs NoRisk")
plt.tight_layout()
plt.show()



#CONFUSION MATRIX


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Create and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["E4−", "E4+"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: APOE E4+ Prediction using BAG")
plt.tight_layout()
plt.show()



# RISK 

#using cbag

#AD vs No risk 


# === Filter data: AD vs NoRisk
df_risk = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()
df_risk['Group'] = (df_risk['Risk'] == 'AD').astype(int)

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np

# === Define true labels and BAG scores
y_true = df_risk['Group'].values
y_score = df_risk['cBAG'].values  # You can switch to df_risk['cBAG'] if desired

# === Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Find best threshold using Youden’s J
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Apply threshold to classify
y_pred = (y_score >= best_threshold).astype(int)

# === Compute performance metrics
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Print results
print(" RISK: AD vs NoRisk using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall (Sensitivity): {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === Plot ROC curve
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
plt.show()

print()

#CHECK WITH METRICS ARE SO GREAT

# Filter AD and No risk
df_subset = df[df['Risk'].isin(['NoRisk', 'AD'])].copy()

# Print
print(df_subset[['MRI_Exam_fixed', 'Risk', 'Age', 'Predicted_Age', 'cBAG']].sort_values('cBAG', ascending=False))

print()


import seaborn as sns
import matplotlib.pyplot as plt

# Violin plot BAG vs group
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_subset, x='Risk', y='cBAG', palette="Set2")
plt.title("cBAG Distribution: AD vs NoRisk")
plt.tight_layout()
plt.show()



#CONFUSION MATRIX


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Create and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["E4−", "E4+"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: APOE E4+ Prediction using cBAG")
plt.tight_layout()
plt.show()








# NoRisk + Familial vs MCI + AD

#BAG


# === Filter data: NoRisk + Familial vs MCI + AD
df_risk = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_risk['Group'] = df_risk['Risk'].isin(['MCI', 'AD']).astype(int)  # 1 if MCI/AD, else 0

# Define true labels and prediction scores
y_true = df_risk['Group'].values
y_score = df_risk['BAG'].values

# ROC, AUC and threshold analysis
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# Metrics
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


# ROC curve
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
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Impairment", "MCI/AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: NoRisk+Familial vs MCI+AD using BAG")
plt.tight_layout()
plt.show()

print()


#Violin plot

import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Filter only the relevant risk groups
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()

# === Step 2: Create a binary group label for plotting
df_plot['Group'] = df_plot['Risk'].apply(lambda x: 'NoRisk+Familial' if x in ['NoRisk', 'Familial'] else 'MCI+AD')

# === Step 3: Plot violin plot of BAG across the two groups
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Group', y='BAG', palette="Set2")
plt.title("BAG Distribution: NoRisk+Familial vs MCI+AD")
plt.tight_layout()
plt.show()





#cBAG



# === Filter data: NoRisk + Familial vs MCI + AD
df_risk = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()
df_risk['Group'] = df_risk['Risk'].isin(['MCI', 'AD']).astype(int)  # 1 if MCI/AD, else 0

# Define true labels and prediction scores
y_true = df_risk['Group'].values
y_score = df_risk['cBAG'].values

# ROC, AUC and threshold analysis
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred = (y_score >= best_threshold).astype(int)

# Metrics
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

# ROC curve
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
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Impairment", "MCI/AD"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: NoRisk+Familial vs MCI+AD using cBAG")
plt.tight_layout()
plt.show()

print()

#Violin plot

import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Filter only the relevant risk groups
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial', 'MCI', 'AD'])].copy()

# === Step 2: Create a binary group label for plotting
df_plot['Group'] = df_plot['Risk'].apply(lambda x: 'NoRisk+Familial' if x in ['NoRisk', 'Familial'] else 'MCI+AD')

# === Step 3: Plot violin plot of cBAG across the two groups
plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Group', y='cBAG', palette="Set2")
plt.title("cBAG Distribution: NoRisk+Familial vs MCI+AD")
plt.tight_layout()
plt.show()


#Sex
#BAG


# === SEX: Female vs Male ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# === Define binary target: 1 = Female, 0 = Male
df['Group'] = (df['sex'] == 'F').astype(int)
y_true = df['Group'].values

# === Use BAG or cBAG as predictor
y_score = df['BAG'].values  # O usa df['cBAG'].values si quieres con el corregido

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Mejor umbral (Youden’s J)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Binarizar predicción
y_pred = (y_score >= best_threshold).astype(int)

# === Métricas de rendimiento
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Imprimir resultados
print(" SEX prediction using BAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === ROC curve
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
plt.show()

# === Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Sex Prediction using BAG")
plt.tight_layout()
plt.show()

print()

#Violin sex all  subjects
import seaborn as sns
import matplotlib.pyplot as plt

# === Create a column with sex labels (optional, just for clarity)
df['Sex_Label'] = df['sex'].map({'F': 'Female', 'M': 'Male'})

# === Violin plot of cBAG by sex
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='Sex_Label', y='BAG', palette="pastel", inner="box")
plt.title("Corrected Brain Age Gap (BAG) by Sex")
plt.xlabel("Sex")
plt.ylabel("BAG")
plt.tight_layout()
plt.show()

#Violin sex only healthy subjects

df_plot = df[df['Risk'].isin(['NoRisk', 'Familial'])].copy()
df_plot['Sex_Label'] = df_plot['sex'].map({'F': 'Female', 'M': 'Male'})

plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Sex_Label', y='BAG', palette="pastel", inner="box")
plt.title("BAG by Sex (NoRisk + Familial only)")
plt.xlabel("Sex")
plt.ylabel("BAG")
plt.tight_layout()
plt.show()



#cBAG


# === SEX: Female vs Male ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# === Define binary target: 1 = Female, 0 = Male
df['Group'] = (df['sex'] == 'F').astype(int)
y_true = df['Group'].values

# === Use BAG or cBAG as predictor
y_score = df['cBAG'].values  # O usa df['cBAG'].values si quieres con el corregido

# === ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc_value = roc_auc_score(y_true, y_score)

# === Mejor umbral (Youden’s J)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

# === Binarizar predicción
y_pred = (y_score >= best_threshold).astype(int)

# === Métricas de rendimiento
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# === Imprimir resultados
print(" SEX prediction using cBAG")
print(f"AUC: {auc_value:.3f}")
print(f"Best threshold (Youden’s J): {best_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === ROC curve
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
plt.show()

# === Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Sex Prediction using cBAG")
plt.tight_layout()
plt.show()

#Violin
import seaborn as sns
import matplotlib.pyplot as plt

# === Create a column with sex labels (optional, just for clarity)
df['Sex_Label'] = df['sex'].map({'F': 'Female', 'M': 'Male'})

# === Violin plot of cBAG by sex
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='Sex_Label', y='cBAG', palette="pastel", inner="box")
plt.title("Corrected Brain Age Gap (cBAG) by Sex")
plt.xlabel("Sex")
plt.ylabel("Corrected BAG")
plt.tight_layout()
plt.show()


#Violin only healthy
df_plot = df[df['Risk'].isin(['NoRisk', 'Familial'])].copy()
df_plot['Sex_Label'] = df_plot['sex'].map({'F': 'Female', 'M': 'Male'})

plt.figure(figsize=(6, 4))
sns.violinplot(data=df_plot, x='Sex_Label', y='cBAG', palette="pastel", inner="box")
plt.title("Corrected BAG by Sex (NoRisk + Familial only)")
plt.xlabel("Sex")
plt.ylabel("Corrected BAG")
plt.tight_layout()
plt.show()
