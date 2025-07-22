# SCATTER 1
# ============================== IMPORTS ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ============================ CONFIGURATION ============================
CSV_PATH = (
    "/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/ADDECODE Saving figures/addecode_training_eval_plots_save/cv_predictions_addecode.csv"
)
output_dir = "ADDECODE_original_scatter_1"
os.makedirs(output_dir, exist_ok=True)

# REAL vs PREDICTED Scatter plots



# === LOAD CSV FILE ===
df = pd.read_csv(CSV_PATH)

# =================================================================================
# 1) SCATTER PLOT — ALL POINTS (one prediction per repetition)
# =================================================================================

# --- Data ---
x_all = df["Real_Age"].values
y_all = df["Predicted_Age"].values

# --- Overall metrics across all points ---
mae_all  = mean_absolute_error(x_all, y_all)
rmse_all = mean_squared_error(x_all, y_all, squared=False)
r2_all   = r2_score(x_all, y_all)

# --- Compute metrics per (fold, repeat) to estimate standard deviation ---
grouped = df.groupby(["Fold", "Repeat"]).apply(
    lambda g: pd.Series({
        "MAE": mean_absolute_error(g["Real_Age"], g["Predicted_Age"]),
        "RMSE": mean_squared_error(g["Real_Age"], g["Predicted_Age"], squared=False),
        "R2": r2_score(g["Real_Age"], g["Predicted_Age"])
    })
)

mae_std, rmse_std, r2_std = grouped["MAE"].std(), grouped["RMSE"].std(), grouped["R2"].std()

# --- Linear regression (trend line) ---
reg = LinearRegression().fit(x_all.reshape(-1, 1), y_all)
slope, intercept = reg.coef_[0], reg.intercept_

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(x_all, y_all, alpha=0.6, label="Predictions", edgecolors="k")

min_val, max_val = min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', alpha=0.6, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

# --- Metrics box in bottom-right ---
plt.gca().text(0.95, 0.05,
    f"MAE = {mae_all:.2f} ± {mae_std:.2f}\n"
    f"RMSE = {rmse_all:.2f} ± {rmse_std:.2f}\n"
    f"R² = {r2_all:.2f} ± {r2_std:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)


plt.xlabel("Real Age (years)")
plt.ylabel("Predicted Age (years)")
plt.title("Scatter Plot — All Repetitions")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_all_predictions.png"), dpi=300)
plt.close()





# =================================================================================
# 2) SCATTER PLOT — MEAN PREDICTION PER SUBJECT
# =================================================================================

# --- Compute mean prediction per subject ---
df_mean = df.groupby("Subject_ID").agg({
    "Real_Age": "first",
    "Predicted_Age": "mean"
}).reset_index()

x_mean = df_mean["Real_Age"].values
y_mean = df_mean["Predicted_Age"].values

# --- Metrics ---
mae_mean  = mean_absolute_error(x_mean, y_mean)
rmse_mean = mean_squared_error(x_mean, y_mean, squared=False)
r2_mean   = r2_score(x_mean, y_mean)

# --- Linear regression (trend line) ---
reg = LinearRegression().fit(x_mean.reshape(-1, 1), y_mean)
slope, intercept = reg.coef_[0], reg.intercept_

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(x_mean, y_mean, alpha=0.7, label="Mean per Subject", edgecolors="k")

min_val, max_val = min(x_mean.min(), y_mean.min()), max(x_mean.max(), y_mean.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', alpha=0.6, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

# --- Metrics box in bottom-right ---
plt.gca().text(0.95, 0.05,
    f"MAE = {mae_mean:.2f}\n"
    f"RMSE = {rmse_mean:.2f}\n"
    f"R² = {r2_mean:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)


plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter Plot — Mean Prediction per Subject")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_mean_prediction_per_subject.png"), dpi=300)
plt.close()



# =================================================================================
# 3) SCATTER PLOT — MEAN ± SD PER SUBJECT (error bars)
# =================================================================================

# --- Compute mean and std of predictions per subject ---
df_stats = df.groupby("Subject_ID").agg({
    "Real_Age":       "first",          # same real age across repeats
    "Predicted_Age": ["mean", "std"]    # mean and SD of predicted age
}).reset_index()
df_stats.columns = ["Subject_ID", "Real_Age", "Pred_Mean", "Pred_SD"]

x_err = df_stats["Real_Age"].values
y_err = df_stats["Pred_Mean"].values
y_std = df_stats["Pred_SD"].values      # error bar length (±1 SD)

# --- Metrics using the means (same as Plot #2) ---
mae_err  = mean_absolute_error(x_err, y_err)
rmse_err = mean_squared_error(x_err, y_err, squared=False)
r2_err   = r2_score(x_err, y_err)

# --- Linear regression (trend line) ---
reg = LinearRegression().fit(x_err.reshape(-1, 1), y_err)
slope, intercept = reg.coef_[0], reg.intercept_

# --- Plot with error bars ---
plt.figure(figsize=(8, 6))
plt.errorbar(
    x_err, y_err, yerr=y_std,
    fmt='o', ecolor='gray', elinewidth=1, capsize=3,
    markeredgecolor='k', markerfacecolor='tab:blue', alpha=0.8,
    label="Mean ± SD per Subject"
)

min_val, max_val = min(x_err.min(), y_err.min()), max(x_err.max(), y_err.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val],
         reg.predict([[min_val], [max_val]]), 'b-', alpha=0.6, linewidth=2,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

# --- Metrics box in bottom-right (gray background) ---
plt.gca().text(
    0.95, 0.05,
    f"MAE = {mae_err:.2f}\nRMSE = {rmse_err:.2f}\nR² = {r2_err:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)

plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter Plot — Mean ± SD per Subject")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_mean_prediction_per_subject_errorbars.png"), dpi=300)
plt.close()






# 2 and 3 

# ============================================
# Compute mean prediction per subject
# ============================================
df_mean = df.groupby("Subject_ID").agg({
    "Real_Age": "first",
    "Predicted_Age": "mean"
}).reset_index()

x_mean = df_mean["Real_Age"].values
y_mean = df_mean["Predicted_Age"].values

# ============================================
# Compute metrics per (Fold, Repeat)
# ============================================
metrics_by_split = []

for (fold, rep), grp in df.groupby(["Fold", "Repeat"]):
    g_mean = grp.groupby("Subject_ID").agg({
        "Real_Age": "first",
        "Predicted_Age": "mean"
    })
    mae  = mean_absolute_error(g_mean["Real_Age"], g_mean["Predicted_Age"])
    rmse = mean_squared_error(g_mean["Real_Age"], g_mean["Predicted_Age"], squared=False)
    r2   = r2_score(g_mean["Real_Age"], g_mean["Predicted_Age"])
    metrics_by_split.append([mae, rmse, r2])

metrics_by_split = np.array(metrics_by_split)
mae_mean,  mae_std  = metrics_by_split[:, 0].mean(), metrics_by_split[:, 0].std(ddof=1)
rmse_mean, rmse_std = metrics_by_split[:, 1].mean(), metrics_by_split[:, 1].std(ddof=1)
r2_mean,   r2_std   = metrics_by_split[:, 2].mean(), metrics_by_split[:, 2].std(ddof=1)

# ============================================
# Plot 2 — Mean prediction per subject
# ============================================
reg = LinearRegression().fit(x_mean.reshape(-1, 1), y_mean)
slope, intercept = reg.coef_[0], reg.intercept_

plt.figure(figsize=(8, 6))
plt.scatter(x_mean, y_mean, alpha=0.7, label="Mean per Subject", edgecolors="k")

min_val, max_val = min(x_mean.min(), y_mean.min()), max(x_mean.max(), y_mean.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', linewidth=2, alpha=0.6,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

plt.gca().text(
    0.95, 0.05,
    f"MAE  = {mae_mean:.2f} ± {mae_std:.2f}\n"
    f"RMSE = {rmse_mean:.2f} ± {rmse_std:.2f}\n"
    f"R²   = {r2_mean:.2f} ± {r2_std:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)

plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter — Mean Prediction per Subject")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2scatter_mean_prediction_per_subject.png"), dpi=300)
plt.close()

# ============================================
# Plot 3 — Mean ± SD per subject (error bars)
# ============================================
df_stats = df.groupby("Subject_ID").agg({
    "Real_Age": "first",
    "Predicted_Age": ["mean", "std"]
}).reset_index()
df_stats.columns = ["Subject_ID", "Real_Age", "Pred_Mean", "Pred_SD"]

x_err = df_stats["Real_Age"].values
y_err = df_stats["Pred_Mean"].values
y_std = df_stats["Pred_SD"].values

reg = LinearRegression().fit(x_err.reshape(-1, 1), y_err)
slope, intercept = reg.coef_[0], reg.intercept_

plt.figure(figsize=(8, 6))
plt.errorbar(
    x_err, y_err, yerr=y_std,
    fmt='o', ecolor='gray', elinewidth=1, capsize=3,
    markeredgecolor='k', markerfacecolor='tab:blue', alpha=0.8,
    label="Mean ± SD per Subject"
)

min_val, max_val = min(x_err.min(), y_err.min()), max(x_err.max(), y_err.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y = x)")
plt.plot([min_val, max_val], reg.predict([[min_val], [max_val]]), 'b-', linewidth=2, alpha=0.6,
         label=f"Trend: y = {slope:.2f}x + {intercept:.1f}")

plt.gca().text(
    0.95, 0.05,
    f"MAE  = {mae_mean:.2f} ± {mae_std:.2f}\n"
    f"RMSE = {rmse_mean:.2f} ± {rmse_std:.2f}\n"
    f"R²   = {r2_mean:.2f} ± {r2_std:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black', alpha=0.8)
)

plt.xlabel("Real Age (years)")
plt.ylabel("Mean Predicted Age (years)")
plt.title("Scatter — Mean ± SD per Subject")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2scatter_mean_prediction_per_subject_errorbars.png"), dpi=300)
plt.close()