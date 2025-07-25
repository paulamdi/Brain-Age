# ADRC Cognition regressions

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === Load ADRC dataset with BAG and cognitive scores ===
df = pd.read_csv("/home/bas/Desktop/Paula DTI_fMRI Codes/ADRC/BEST/brainage_predictions_adrc_all_clipped120.csv")  



# === Select cognitive columns ===
cognitive_cols = df.columns[62:97].tolist()  # Python uses 0-based indexing

# === Clean base dataframe ===
df_base = df[["BAG", "cBAG"] + cognitive_cols].dropna(subset=["BAG", "cBAG"])

# === Output directory ===
output_dir = "regression_BAG_cognition_ADRC"
os.makedirs(output_dir, exist_ok=True)

# === Initialize results list ===
results = []

for metric in cognitive_cols:
    df_clean = df_base[[metric, "BAG", "cBAG"]].dropna()

    if df_clean.shape[0] < 10:
        continue

    # === Regressions ===
    X_bag = sm.add_constant(df_clean["BAG"])
    y = df_clean[metric]
    model_bag = sm.OLS(y, X_bag).fit()

    X_cbag = sm.add_constant(df_clean["cBAG"])
    model_cbag = sm.OLS(y, X_cbag).fit()

    # === Append regression results ===
    results.append({
        "Metric": metric,
        "R²_BAG": model_bag.rsquared,
        "p_BAG": model_bag.pvalues[1],
        "Coef_BAG": model_bag.params[1],
        "R²_cBAG": model_cbag.rsquared,
        "p_cBAG": model_cbag.pvalues[1],
        "Coef_cBAG": model_cbag.params[1]
    })

    # === Plot and save figure ===
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    sns.regplot(x="BAG", y=metric, data=df_clean, ax=axes[0])
    axes[0].set_title(f"{metric} vs. BAG")
    # === Annotate BAG plot ===
    # === Annotate BAG plot ===
    axes[0].text(

        0.65, 0.2,

        f"p-value = {model_bag.pvalues[1]:.3g}\n"
        f"R² = {model_bag.rsquared:.2f}\n"
        f"β = {model_bag.params[1]:.2f}",
        transform=axes[0].transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )


        
      
    sns.regplot(x="cBAG", y=metric, data=df_clean, ax=axes[1])
    axes[1].set_title(f"{metric} vs. cBAG")
    # === Annotate cBAG plot ===
    axes[1].text(
        0.65, 0.2,
        f"p-value = {model_cbag.pvalues[1]:.3g}\n"
        f"R² = {model_cbag.rsquared:.2f}\n"
        f"β = {model_cbag.params[1]:.2f}",
        transform=axes[1].transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )


    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{metric}_regression.png")
    plt.savefig(fig_path)
    plt.close()

# === Save regression results CSV ===
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by="R²_BAG", ascending=False)
csv_path = os.path.join(output_dir, "regression_results_BAG_cognition_ADRC.csv")
df_results_sorted.to_csv(csv_path, index=False)

print(f"Saved: {csv_path}")
print(f"Saved plots to: {output_dir}")
