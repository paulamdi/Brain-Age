
# === Script: Generate BAG and cBAG vs Age plots ===

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import os

# === Load dataset ===
df = pd.read_csv("BAG_with_all_metadata.csv")
df_preds = df.copy()  # if you want to modify safely


df["Predicted_Age"] = df["Predicted_Age"]  
df["BAG_check"] = df["Predicted_Age"] - df["Age"]
print((df["BAG"] - df["BAG_check"]).abs().max())


# === Create output folder ===
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# === Function to plot and save ===
def plot_bag_relation(df, y_var, color, title_suffix, save_name):
    X = df[["Age"]]
    y = df[y_var]
    
    # Linear regression and stats
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    r, p = pearsonr(df["Age"], y)
    
    # Plot
    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(x="Age", y=y_var, data=df, alpha=0.7)
    sns.regplot(x="Age", y=y_var, data=df, scatter=False, color=color, label="Trend")
    plt.axhline(0, linestyle="--", color="gray", lw=1)
    
    plt.title(f"{y_var} vs Age – {title_suffix}", fontsize=13)
    plt.xlabel("Chronological Age (years)", fontsize=11)
    plt.ylabel(y_var, fontsize=11)

    metrics_txt = (
        f"Slope = {slope:+.2f}\n"
        f"R = {r:+.2f}\n"
        f"p = {p:.1e}"
    )
    
    plt.text(0.05, 0.95, metrics_txt,
             transform=plt.gca().transAxes,
             ha="left", va="top",
             fontsize=10,
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_name), dpi=300)
    plt.close()

# === Run for BAG and cBAG ===
plot_bag_relation(df_preds, y_var="BAG", color="red", title_suffix="AD-DECODE", save_name="ADDECODE_BAG_vs_Age.png")
plot_bag_relation(df_preds, y_var="cBAG", color="green", title_suffix="AD-DECODE", save_name="ADDECODE_cBAG_vs_Age.png")

print("✅ Saved BAG and cBAG vs Age plots to /plots")
