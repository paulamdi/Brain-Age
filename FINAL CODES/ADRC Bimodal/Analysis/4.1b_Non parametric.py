# Kruskal-Wallis and Mann–Whitney U Tests for BAG and cBAG Across Groups

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import itertools
import os

# === Load your data from CSV ===
df = pd.read_csv("brainage_predictions_adrc_all_clipped120.csv")  

# === Define cognitive risk group based on diagnosis ===
def assign_risk(row):
    if "DEMENTED" in row and row["DEMENTED"] == 1:
        return "Demented"
    elif "IMPNOMCI" in row and row["IMPNOMCI"] == 1:
        return "MCI"
    elif "NORMCOG" in row and row["NORMCOG"] == 1:
        return "NoRisk"
    else:
        return None

# Apply risk categorization to dataframe
df["Risk"] = df.apply(assign_risk, axis=1)

# === Add APOE status and sex if missing ===
if "APOE_status" not in df.columns:
    def get_apoe_status(apoe):
        if apoe in ["3/4", "4/4"]:
            return "E4+"
        elif apoe in ["2/3", "3/3"]:
            return "E4-"
        else:
            return "Unknown"
    df["APOE_status"] = df["APOE"].apply(get_apoe_status)

if "sex" not in df.columns:
    df["sex"] = df["SUBJECT_SEX"].map({1: "M", 2: "F"})

# === Define groups and metrics to test ===
group_vars = {
    "Risk": ["NoRisk", "MCI", "Demented"],
    "APOE": ["2/3", "3/3", "3/4", "4/4"],
    "APOE_status": ["E4-", "E4+"],
    "sex": ["F", "M"]
}
metrics = ["BAG", "cBAG"]

# === Create output directory ===
output_dir = "nonparametric_stats_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Function to convert p-values into significance stars ===
def format_p_value(p):
    if p <= 1e-4: return "****"
    elif p <= 1e-3: return "***"
    elif p <= 1e-2: return "**"
    elif p <= 5e-2: return "*"
    else: return "ns"

# === Loop over each metric (BAG, cBAG) ===
for metric in metrics:
    results = []

    # === Loop over each grouping variable (Risk, APOE...) ===
    for var, groups in group_vars.items():
        # Extract values per group (dropping NaNs)
        data_groups = [df[df[var] == g][metric].dropna() for g in groups if g in df[var].unique()]
        if len(data_groups) > 1:
            # Perform Kruskal-Wallis (global nonparametric comparison)
            stat, p_kw = kruskal(*data_groups)
            results.append({
                "Metric": metric,
                "Variable": var,
                "Comparison": "Global",
                "Test": "Kruskal-Wallis",
                "p-value": p_kw,
                "Significance": format_p_value(p_kw)
            })

        # === Pairwise Mann–Whitney U tests ===
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

    # === Save results to CSV for each metric ===
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"nonparametric_results_{metric}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
