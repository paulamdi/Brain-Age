import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal
from scipy.stats import mannwhitneyu


# === Output folder for all plots ===
plot_dir = "violins_addecode"
os.makedirs(plot_dir, exist_ok=True)




# VIOLIN PLOT BAG-RISK



# === Load the data
df = pd.read_csv("brain_age_predictions_with_metadata.csv")




# ------------------------------------------------------------
# Violin: BAG vs Risk Group + Kruskal-Wallis p-value
# ------------------------------------------------------------

# --- Define order of risk groups ---
risk_order = ["NoRisk", "Familial", "MCI", "AD"]

# --- Kruskal-Wallis test (BAG across Risk groups) ---
groups_risk = [df[df["Risk"] == r]["BAG"].dropna() for r in risk_order]
stat_r, pval_r = kruskal(*groups_risk)

# --- Format p-value ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

star_r = format_p_value(pval_r)
pval_text_r = f"Kruskal-Wallis: p = {pval_r:.3g} ({star_r})"

# --- Plot ---
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df, x="Risk", y="BAG",
    order=risk_order, inner="box", palette="Set2"
)
plt.title("Brain Age Gap (BAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Brain Age Gap (BAG)")

# --- Add p-value box ---
plt.text(
    0.02, 0.97, pval_text_r,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)


# --- Save ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_BAG_vs_Risk_with_pvalue.png"), dpi=300)
plt.close()






# ------------------------------------------------------------
# Violin: cBAG vs Risk group  +  global Kruskal-Wallis p-value
# ------------------------------------------------------------

# --- Order of groups ---
risk_order = ["NoRisk", "Familial", "MCI", "AD"]

# --- Kruskal-Wallis test (cBAG across Risk) ---
groups_cbag = [df[df["Risk"] == g]["cBAG"].dropna() for g in risk_order]
stat_c, pval_c = kruskal(*groups_cbag)

# --- Convert p-value to stars ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

star_c = format_p_value(pval_c)
pval_text_c = f"Kruskal-Wallis: p = {pval_c:.3g} ({star_c})"

# --- Plot ---
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df, x="Risk", y="cBAG",
    order=risk_order, inner="box", palette="Set2"
)
plt.title("Corrected Brain Age Gap (cBAG) by Risk Group")
plt.xlabel("Risk Group")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# --- Add p-value label (upper-left corner) ---

plt.text(
    0.02, 0.97, pval_text_c,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_cBAG_vs_Risk_with_pvalue.png"), dpi=300)
plt.close()



#GENOTYPE



# ------------------------------------------------------------
# Violin: BAG vs APOE Genotype  +  global Kruskal-Wallis p-value
# ------------------------------------------------------------

# --- Map numeric genotype codes to labels ---
genotype_labels = {0: "APOE23", 1: "APOE33", 2: "APOE34", 3: "APOE44"}
df["genotype_label"] = df["genotype"].map(genotype_labels)

# --- Order of genotype categories ---
genotype_order = ["APOE23", "APOE33", "APOE34", "APOE44"]

# --- Global Kruskal-Wallis test (BAG across genotypes) ---
group_bag_gen = [df[df["genotype_label"] == g]["BAG"].dropna() for g in genotype_order]
stat_gen, pval_gen = kruskal(*group_bag_gen)

# --- Helper: format p-value into stars ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars_gen = format_p_value(pval_gen)
pval_text_gen = f"Kruskal-Wallis: p = {pval_gen:.3g} ({stars_gen})"

# --- Plot violin ---
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="genotype_label", y="BAG",
    order=genotype_order,
    inner="box", palette="pastel"
)
plt.title("Brain Age Gap (BAG) by APOE Genotype")
plt.xlabel("APOE Genotype")
plt.ylabel("Brain Age Gap (BAG)")

# --- Add p-value box (upper-left corner) ---

plt.text(
    0.02, 0.97, pval_text_gen,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_BAG_vs_Genotype_with_pvalue.png"), dpi=300)
plt.close()







# ------------------------------------------------------------
# Violin: cBAG vs APOE Genotype  +  global Kruskal-Wallis p-value
# ------------------------------------------------------------

# --- Map numeric genotype codes to labels ---
genotype_labels = {0: "APOE23", 1: "APOE33", 2: "APOE34", 3: "APOE44"}
df["genotype_label"] = df["genotype"].map(genotype_labels)

# --- Order of genotype categories ---
genotype_order = ["APOE23", "APOE33", "APOE34", "APOE44"]

# --- Global Kruskal-Wallis test (cBAG across genotypes) ---
groups_cbag_gen = [
    df[df["genotype_label"] == g]["cBAG"].dropna()
    for g in genotype_order
]
stat_cg, pval_cg = kruskal(*groups_cbag_gen)

# --- Helper: format p-value into stars ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars_cg   = format_p_value(pval_cg)
pval_text_cg = f"Kruskal-Wallis: p = {pval_cg:.3g} ({stars_cg})"

# --- Plot violin ---
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df,
    x="genotype_label", y="cBAG",
    order=genotype_order,
    inner="box", palette="pastel"
)
plt.title("Corrected Brain Age Gap (cBAG) by APOE Genotype")
plt.xlabel("APOE Genotype")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# --- Add p-value box (upper-left corner) ---

plt.text(
    0.02, 0.97, pval_text_cg,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_cBAG_vs_Genotype_with_pvalue.png"), dpi=300)
plt.close()








# VIOLIN PLOT  APOE E4+ E4-




# ------------------------------------------------------------
# Violin: BAG vs APOE status (E4- vs E4+)  +  Mann-Whitney p-value
# ------------------------------------------------------------

# --- Define display order ---
apoe_order = ["E4-", "E4+"]

# --- Extract data for both groups ---
bag_e4_neg = df[df["APOE"] == "E4-"]["BAG"].dropna()
bag_e4_pos = df[df["APOE"] == "E4+"]["BAG"].dropna()

# --- Mann-Whitney U Test ---
stat_apoe_bag, pval_apoe_bag = mannwhitneyu(bag_e4_neg, bag_e4_pos, alternative="two-sided")

# --- Format significance level ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars = format_p_value(pval_apoe_bag)
pval_text = f"Mann-Whitney U: p = {pval_apoe_bag:.3g} ({stars})"

# --- Create violin plot ---
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="APOE", y="BAG", order=apoe_order, inner="box", palette="pastel")
plt.title("Brain Age Gap (BAG) by APOE Risk Status")
plt.xlabel("APOE Status")
plt.ylabel("Brain Age Gap (BAG)")

# --- Add p-value annotation ---

plt.text(
    0.02, 0.97, pval_text,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_BAG_vs_GenotypeStatus_with_pvalue.png"), dpi=300)
plt.close()



# ------------------------------------------------------------
# Violin: cBAG vs APOE status (E4- vs E4+)  +  Mann-Whitney p-value
# ------------------------------------------------------------

# --- Define order ---
apoe_order = ["E4-", "E4+"]

# --- Extract data for both groups ---
cbag_e4_neg = df[df["APOE"] == "E4-"]["cBAG"].dropna()
cbag_e4_pos = df[df["APOE"] == "E4+"]["cBAG"].dropna()

# --- Mann-Whitney U Test ---
stat_apoe_cbag, pval_apoe_cbag = mannwhitneyu(cbag_e4_neg, cbag_e4_pos, alternative="two-sided")

# --- Format significance level ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars = format_p_value(pval_apoe_cbag)
pval_text = f"Mann-Whitney U: p = {pval_apoe_cbag:.3g} ({stars})"

# --- Create violin plot ---
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x="APOE", y="cBAG", order=apoe_order, inner="box", palette="pastel")
plt.title("Corrected Brain Age Gap (cBAG) by APOE Risk Status")
plt.xlabel("APOE Status")
plt.ylabel("Corrected Brain Age Gap (cBAG)")

# --- Add p-value annotation box ---

plt.text(
    0.02, 0.97, pval_text,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_cBAG_vs_GenotypeStatus_with_pvalue.png"), dpi=300)
plt.close()








#VIOLIN SEX



# ------------------------------------------------------------
# Violin: BAG by Sex (F vs M)  +  Mann-Whitney U test
# ------------------------------------------------------------

# --- Filter valid values ---
df = df[df["sex"].isin(["M", "F"])].copy()
df["sex"] = pd.Categorical(df["sex"], categories=["F", "M"], ordered=True)

# --- Extract data for each sex ---
bag_f = df[df["sex"] == "F"]["BAG"].dropna()
bag_m = df[df["sex"] == "M"]["BAG"].dropna()

# --- Mann-Whitney U test ---
stat_sex_bag, pval_sex_bag = mannwhitneyu(bag_f, bag_m, alternative="two-sided")

# --- Format p-value with stars ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars = format_p_value(pval_sex_bag)
pval_text = f"Mann-Whitney U: p = {pval_sex_bag:.3g} ({stars})"

# --- Create violin plot ---
plt.figure(figsize=(6, 5))
sns.violinplot(data=df, x="sex", y="BAG", inner="box", palette="Set2")
plt.title("BAG by Sex (AD-DECODE)")
plt.xlabel("Sex")
plt.ylabel("Brain Age Gap (BAG)")
plt.grid(True)

# --- Add p-value annotation ---

plt.text(
    0.02, 0.97, pval_text,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_BAG_vs_Sex_with_pvalue.png"), dpi=300)
plt.close()






# ------------------------------------------------------------
# Violin: cBAG by Sex (F vs M)  +  Mann-Whitney U test
# ------------------------------------------------------------

# --- Filter valid values ---
df = df[df["sex"].isin(["M", "F"])].copy()
df["sex"] = pd.Categorical(df["sex"], categories=["F", "M"], ordered=True)

# --- Extract data for each sex ---
cbag_f = df[df["sex"] == "F"]["cBAG"].dropna()
cbag_m = df[df["sex"] == "M"]["cBAG"].dropna()

# --- Mann-Whitney U test ---
stat_sex_cbag, pval_sex_cbag = mannwhitneyu(cbag_f, cbag_m, alternative="two-sided")

# --- Format p-value with stars ---
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

stars = format_p_value(pval_sex_cbag)
pval_text = f"Mann-Whitney U: p = {pval_sex_cbag:.3g} ({stars})"

# --- Create violin plot ---
plt.figure(figsize=(6, 5))
sns.violinplot(data=df, x="sex", y="cBAG", inner="box", palette="Set2")
plt.title("Corrected BAG (cBAG) by Sex (AD-DECODE)")
plt.xlabel("Sex")
plt.ylabel("Corrected Brain Age Gap (cBAG)")
plt.grid(True)

# --- Add p-value annotation ---

# --- Add p-value box ---
plt.text(
    0.02, 0.97, pval_text,  
    transform=plt.gca().transAxes,
    ha="left", va="top",
    fontsize=12,
    fontweight="bold",  
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="black")
)

# --- Save figure ---
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "Violin_cBAG_vs_Sex_with_pvalue.png"), dpi=300)
plt.close()







#Pvalue

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import itertools

# === Load your CSV ===
df = pd.read_csv("BAG_with_all_metadata.csv")

# === Optional: create readable genotype label ===
genotype_labels = {0: "APOE23", 1: "APOE33", 2: "APOE34", 3: "APOE44"}
df["genotype_label"] = df["genotype"].map(genotype_labels)

# === Helper to format p-values ===
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

# === Define group variables and levels ===
group_vars = {
    "Risk": ["NoRisk", "Familial", "MCI", "AD"],
    "genotype_label": ["APOE23", "APOE33", "APOE34", "APOE44"],
    "APOE": ["E4-", "E4+"],
    "sex": ["F", "M"]
}

metrics = ["BAG", "cBAG"]

# === Loop through metrics and group variables ===
for metric in metrics:
    results = []

    for var, groups in group_vars.items():
        # Global Kruskal-Wallis (if more than 2 groups)
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

        # Pairwise Mann-Whitney U
        for g1, g2 in itertools.combinations(groups, 2):
            if g1 in df[var].unique() and g2 in df[var].unique():
                data1 = df[df[var] == g1][metric].dropna()
                data2 = df[df[var] == g2][metric].dropna()
                if len(data1) > 0 and len(data2) > 0:
                    stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
                    results.append({
                        "Metric": metric,
                        "Variable": var,
                        "Comparison": f"{g1} vs. {g2}",
                        "Test": "Mann-Whitney U",
                        "p-value": p,
                        "Significance": format_p_value(p)
                    })

    # Save results
   
    # === Output folder for violin plots and stats ===
    plot_dir = "violins_addecode"
    os.makedirs(plot_dir, exist_ok=True)
    
    # === Save results inside the same folder ===
    results_df = pd.DataFrame(results)
    filename = f"stat_results1_{metric}.csv"
    filepath = os.path.join(plot_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

    
    
    
  

    