import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import itertools

# === Load predictions with metadata ===
df = pd.read_csv("brainage_predictions_adrc_all_clipped120.csv")

# === Output directory for saving figures ===
output_dir = "figures_bag_cbags_VIOLIN PLOTS"
os.makedirs(output_dir, exist_ok=True)

# === BAG vs Age ===
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Age", y="BAG", alpha=0.6)
sns.regplot(data=df, x="Age", y="BAG", scatter=False, color="red", label="Trend")
plt.axhline(0, linestyle="--", color="gray")
plt.title("BAG vs Age (Before Correction)")
plt.xlabel("Chronological Age")
plt.ylabel("Brain Age Gap (BAG)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "BAG_vs_Age.png"))
plt.show()

# === cBAG vs Age ===
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Age", y="cBAG", alpha=0.6)
sns.regplot(data=df, x="Age", y="cBAG", scatter=False, color="green", label="Trend")
plt.axhline(0, linestyle="--", color="gray")
plt.title("Corrected BAG vs Age (After Correction)")
plt.xlabel("Chronological Age")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cBAG_vs_Age.png"))
plt.show()


# 2) Derive new categorical columns (Risk, APOE_status, sex) -----
# === Define risk group ===
def assign_risk(row):
    if row["DEMENTED"] == 1:
        return "Demented"
    elif row["DEMENTED"] == 0 and row["IMPNOMCI"] == 1:
        return "MCI"
    elif row["NORMCOG"] == 1:
        return "NoRisk"
    else:
        return None


df["Risk"] = df.apply(assign_risk, axis=1)
df = df[df["Risk"].isin(["NoRisk", "MCI", "Demented"])].copy()



def get_apoe_status(apoe):
    if apoe in ["3/4", "4/4"]:
        return "E4+"
    elif apoe in ["2/3", "3/3"]:
        return "E4-"
    else:
        return "Unknown"

df["APOE_status"] = df["APOE"].apply(get_apoe_status)
df_apoe_status = df[df["APOE_status"].isin(["E4-", "E4+"])].copy()




df["sex"] = df["SUBJECT_SEX"].map({1: "M", 2: "F"})
df_sex = df[df["sex"].isin(["F", "M"])].copy()




# 3) === P-VALUE TESTS
# === P-VALUE TESTS ===
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

group_vars = {
    "Risk": df["Risk"].dropna().unique().tolist(),
    "APOE": df["APOE"].dropna().unique().tolist(),
    "APOE_status": ["E4-", "E4+"],
    "sex": ["F", "M"]
}

metrics = ["BAG", "cBAG"]

for metric in metrics:
    results = []

    for var, groups in group_vars.items():
        group_data = [df[df[var] == g][metric].dropna() for g in groups if g in df[var].unique()]
        if len(group_data) > 1:
            stat, p_kw = kruskal(*group_data)
            results.append({
                "Metric": metric,
                "Variable": var,
                "Comparison": "Global",
                "Test": "Kruskal-Wallis",
                "p-value": p_kw,
                "Significance": format_p_value(p_kw)
            })

        for g1, g2 in itertools.combinations(groups, 2):
            if g1 in df[var].unique() and g2 in df[var].unique():
                d1 = df[df[var] == g1][metric].dropna()
                d2 = df[df[var] == g2][metric].dropna()
                if len(d1) > 0 and len(d2) > 0:
                    stat, p = mannwhitneyu(d1, d2, alternative="two-sided")
                    results.append({
                        "Metric": metric,
                        "Variable": var,
                        "Comparison": f"{g1} vs. {g2}",
                        "Test": "Mann-Whitney U",
                        "p-value": p,
                        "Significance": format_p_value(p)
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"stat_results_{metric}.csv", index=False)
    print(f"Saved: stat_results_{metric}.csv")



# === Save global p-values in dictionary for plotting ===
    if metric == "BAG":
        stat_bag_df = results_df.copy()
    elif metric == "cBAG":
        stat_cbag_df = results_df.copy()

# Create p-value dictionary
pvals_dict = {}
for df_stat, metric in zip([stat_bag_df, stat_cbag_df], ["BAG", "cBAG"]):
    for var in group_vars:
        row = df_stat[(df_stat["Variable"] == var) & (df_stat["Comparison"] == "Global")]
        if not row.empty:
            pvals_dict[(metric, var)] = row["p-value"].values[0]



#VIOLIN PLOTS


# === VIOLIN PLOTS: BAG/cBAG by Risk ===

# === VIOLIN PLOT — BAG by Risk Group ===
plt.figure(figsize=(8, 5))

# Violin plot
sns.violinplot(
    data=df,
    x="Risk",
    y="BAG",
    order=["NoRisk", "MCI", "Demented"],
    inner="box",
    palette="Set2"
)

# Title and labels
plt.title("Brain Age Gap (BAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Brain Age Gap (BAG)")

# Add grid (light, dashed)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add p-value box (Kruskal-Wallis)
pval = pvals_dict.get(("BAG", "Risk"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.35,
        y=df["BAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin bag Risk.png"))
plt.show()



# === VIOLIN PLOT — cBAG by Risk Group ===
plt.figure(figsize=(8, 5))

# Violin plot
sns.violinplot(
    data=df,
    x="Risk",
    y="cBAG",
    order=["NoRisk", "MCI", "Demented"],
    inner="box",
    palette="Set2"
)

# Title and axis labels
plt.title("Corrected Brain Age Gap (cBAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# Add grid (light dashed lines for readability)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add p-value annotation box (Kruskal-Wallis)
pval = pvals_dict.get(("cBAG", "Risk"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.35,
        y=df["cBAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin cbag Risk.png"))
plt.show()





# === VIOLIN PLOTS: BAG/cBAG by APOE ===

# === VIOLIN PLOT — BAG by APOE Genotype ===
apoe_order = ["2/3", "3/3", "3/4", "4/4"]
df_apoe = df[df["APOE"].isin(apoe_order)].copy()

plt.figure(figsize=(8, 5))

# Violin plot
sns.violinplot(
    data=df_apoe,
    x="APOE",
    y="BAG",
    order=apoe_order,
    inner="box",
    palette="pastel"
)

# Title and labels
plt.title("Brain Age Gap (BAG) by APOE Genotype")
plt.xlabel("APOE Genotype")
plt.ylabel("Brain Age Gap (BAG)")

# Add grid for better readability
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add formatted p-value (Kruskal-Wallis)
pval = pvals_dict.get(("BAG", "APOE"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.4,
        y=df_apoe["BAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin bag APOE.png"))
plt.show()


# === VIOLIN PLOT — cBAG by APOE Genotype ===
plt.figure(figsize=(8, 5))

# Violin plot
sns.violinplot(
    data=df_apoe,
    x="APOE",
    y="cBAG",
    order=apoe_order,
    inner="box",
    palette="pastel"
)

# Title and labels
plt.title("Corrected Brain Age Gap (cBAG) by APOE Genotype")
plt.xlabel("APOE Genotype")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# Add light grid for clarity
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add formatted p-value (Kruskal-Wallis)
pval = pvals_dict.get(("cBAG", "APOE"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.4,
        y=df_apoe["cBAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin cbag APOE.png"))
plt.show()







# === VIOLIN PLOTS: APOE status (E4+ vs E4-) ===

# === VIOLIN PLOT — BAG by APOE ε4 Status (E4− vs E4+) ===
plt.figure(figsize=(7, 5))

# Violin plot
sns.violinplot(
    data=df_apoe_status,
    x="APOE_status",
    y="BAG",
    order=["E4-", "E4+"],
    inner="box",
    palette="pastel"
)

# Title and labels
plt.title("Brain Age Gap (BAG) by APOE ε4 Status")
plt.xlabel("APOE Status")
plt.ylabel("Brain Age Gap (BAG)")

# Light dashed grid for clarity
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# --- Add formatted p-value box (Kruskal-Wallis) ---
pval = pvals_dict.get(("BAG", "APOE_status"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.25,                                 # slightly left of first violin
        y=df_apoe_status["BAG"].max() * 0.95,    # 95 % of axis height
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white",
                  edgecolor="black",
                  boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin bag E4+-.png"))
plt.show()




# === VIOLIN PLOT — cBAG by APOE ε4 Status (E4− vs E4+) ===
plt.figure(figsize=(7, 5))

# Violin plot
sns.violinplot(
    data=df_apoe_status,
    x="APOE_status",
    y="cBAG",
    order=["E4-", "E4+"],
    inner="box",
    palette="pastel"
)

# Title and axis labels
plt.title("Corrected Brain Age Gap (cBAG) by APOE ε4 Status")
plt.xlabel("APOE Status")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# Add light dashed grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add p-value box with formatted output
pval = pvals_dict.get(("cBAG", "APOE_status"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.25,
        y=df_apoe_status["cBAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white",
                  edgecolor="black",
                  boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin cbag E4+-.png"))
plt.show()






# === VIOLIN PLOTS: Sex ===

# === VIOLIN PLOT — BAG by Sex (F vs M) ===
plt.figure(figsize=(6, 5))

# Violin plot
sns.violinplot(
    data=df_sex,
    x="sex",
    y="BAG",
    inner="box",
    palette="Set2"
)

# Title and axis labels
plt.title("Brain Age Gap (BAG) by Sex")
plt.xlabel("Sex")
plt.ylabel("Brain Age Gap (BAG)")

# Add light dashed grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add formatted p-value annotation
pval = pvals_dict.get(("BAG", "sex"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.2,
        y=df_sex["BAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin bag sex.png"))
plt.show()



# === VIOLIN PLOT — cBAG by Sex (F vs M) ===
plt.figure(figsize=(6, 5))

# Violin plot
sns.violinplot(
    data=df_sex,
    x="sex",
    y="cBAG",
    inner="box",
    palette="Set2"
)

# Title and axis labels
plt.title("Corrected Brain Age Gap (cBAG) by Sex")
plt.xlabel("Sex")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# Add light dashed grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Add formatted p-value box
pval = pvals_dict.get(("cBAG", "sex"), None)
sig = format_p_value(pval) if pval is not None else "ns"

if pval is not None:
    plt.text(
        x=-0.2,
        y=df_sex["cBAG"].max() * 0.95,
        s=f"Kruskal-Wallis: p = {pval:.3f} ({sig})",
        fontsize=12,
        weight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4")
    )

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Violin cbag sex.png"))
plt.show()

