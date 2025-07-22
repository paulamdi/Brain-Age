# ADDECODE
# === BAG–Cognition Regression (AD-DECODE) =====================================
# Every line is commented in English, as requested.

import os                              # File-system operations
import pandas as pd                    # Data handling
import seaborn as sns                  # Nice statistical plots
import matplotlib.pyplot as plt        # Core plotting library
import statsmodels.api as sm           # Ordinary Least Squares (OLS) regression

# ------------------------------- CONFIG ---------------------------------------
OUT_DIR = "BAG_Cognition_Regression"   # Folder where everything will be stored
os.makedirs(OUT_DIR, exist_ok=True)    # Create folder if it does not exist

CSV_INPUT  = "BAG_with_all_metadata.csv"  # Input file with BAG, cBAG + cognition
CSV_OUTPUT = os.path.join(OUT_DIR, "regression_results_BAG_cognition.csv")

# ------------------------- 1) LOAD & PREPARE DATA -----------------------------
df = pd.read_csv(CSV_INPUT)                                # Load full dataframe
cognitive_cols = df.columns[19:50].tolist()                # Cols 20-50 → cognition
df_base = df[["BAG", "cBAG"] + cognitive_cols].copy()      # Keep only needed cols

# -------------------------- 2) LOOP OVER METRICS ------------------------------
results = []                                               # Collect stats here

for metric in cognitive_cols:
    # --- Clean rows: drop NA in BAG, cBAG or current metric
    df_clean = df_base[["BAG", "cBAG", metric]].dropna()
    if df_clean.shape[0] < 10:                             # Skip if <10 samples
        continue

    # --- Regressions ----------------------------------------------------------
    # BAG model
    X_bag = sm.add_constant(df_clean["BAG"])               # Add intercept term
    y      = df_clean[metric]
    model_bag = sm.OLS(y, X_bag).fit()

    # cBAG model
    X_cbag    = sm.add_constant(df_clean["cBAG"])
    model_cbag = sm.OLS(y, X_cbag).fit()

    # --- Save statistics for summary CSV --------------------------------------
    results.append({
        "Metric"   : metric,
        "R2_BAG"   : model_bag.rsquared,
        "p_BAG"    : model_bag.pvalues[1],
        "Beta_BAG" : model_bag.params[1],
        "R2_cBAG"  : model_cbag.rsquared,
        "p_cBAG"   : model_cbag.pvalues[1],
        "Beta_cBAG": model_cbag.params[1]
    })

    # ----------------------- 3) PLOT & SAVE FIGURE ----------------------------
    pretty_metric = metric.replace("_", " ").title()       # Professional label

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    # --- Plot BAG regression --------------------------------------------------
    sns.regplot(x="BAG", y=metric, data=df_clean, ax=axes[0])
    axes[0].set_title(f"{pretty_metric} vs BAG")
    axes[0].set_xlabel("Brain Age Gap (BAG)")
    axes[0].set_ylabel(pretty_metric)
    axes[0].text(
        0.05, 0.95,
        f"β = {model_bag.params[1]:.3f}\nR² = {model_bag.rsquared:.3f}\np = {model_bag.pvalues[1]:.3g}",
        transform=axes[0].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

    # --- Plot cBAG regression -------------------------------------------------
    sns.regplot(x="cBAG", y=metric, data=df_clean, ax=axes[1])
    axes[1].set_title(f"{pretty_metric} vs cBAG")
    axes[1].set_xlabel("Corrected Brain Age Gap (cBAG)")
    axes[1].set_ylabel(pretty_metric)
    axes[1].text(
        0.05, 0.95,
        f"β = {model_cbag.params[1]:.3f}\nR² = {model_cbag.rsquared:.3f}\np = {model_cbag.pvalues[1]:.3g}",
        transform=axes[1].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

    # --- Save figure in output folder ----------------------------------------
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"{metric}_BAG_cBAG_regression.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

# -------------------------- 4) SAVE SUMMARY CSV -------------------------------
df_results = pd.DataFrame(results)
df_results.sort_values("R2_BAG", ascending=False).to_csv(CSV_OUTPUT, index=False)
print(f"Saved summary: {CSV_OUTPUT}")
print(f"All figures & CSV are in: {OUT_DIR}")
