# ADDECODE

    #BAG - cognition Regression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === Load your dataset ===
df = pd.read_csv("/home/bas/Desktop/Paula/GATS/Better/BEST2/PCA genes/BAG/BAG_with_all_metadata.csv")

# === Select cognitive columns (cols 20 to 50 inclusive) ===
cognitive_cols = df.columns[19:50].tolist()

# === Clean base dataframe ===
df_base = df[["BAG", "cBAG"] + cognitive_cols].dropna(subset=["BAG", "cBAG"])

# === Initialize list to collect regression results ===
results = []

for metric in cognitive_cols:
    df_clean = df_base[[metric, "BAG", "cBAG"]].dropna()

    if df_clean.shape[0] < 10:
        continue

    # === BAG regression ===
    X_bag = sm.add_constant(df_clean["BAG"])
    y = df_clean[metric]
    model_bag = sm.OLS(y, X_bag).fit()

    # === cBAG regression ===
    X_cbag = sm.add_constant(df_clean["cBAG"])
    model_cbag = sm.OLS(y, X_cbag).fit()

    # === Append to results list ===
    results.append({
        "Metric": metric,
        "R²_BAG": model_bag.rsquared,
        "p_BAG": model_bag.pvalues[1],
        "Coef_BAG": model_bag.params[1],
        "R²_cBAG": model_cbag.rsquared,
        "p_cBAG": model_cbag.pvalues[1],
        "Coef_cBAG": model_cbag.params[1]
    })

   
    # === Plot side-by-side regressions with p-values inside the plots ===
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    
    # Plot for BAG
    sns.regplot(x="BAG", y=metric, data=df_clean, ax=axes[0])
    axes[0].set_title(f"{metric} ~ BAG")
    axes[0].text(
        0.05, 0.95,
        f"p = {model_bag.pvalues[1]:.3g}",
        transform=axes[0].transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )
    
    # Plot for cBAG
    sns.regplot(x="cBAG", y=metric, data=df_clean, ax=axes[1])
    axes[1].set_title(f"{metric} ~ cBAG")
    axes[1].text(
        0.05, 0.95,
        f"p = {model_cbag.pvalues[1]:.3g}",
        transform=axes[1].transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
    )
    
    plt.tight_layout()
    plt.show()


# === Convert results to DataFrame and save ===
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by="R²_BAG", ascending=False)
df_results_sorted.to_csv("regression_results_BAG_cognition.csv", index=False)
print("Saved: regression_results_BAG_cognition.csv")
