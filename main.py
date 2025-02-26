import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from itertools import combinations
from statsmodels.stats.anova import AnovaRM  # For repeated measures ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load Excel file (excluding "NMP" and "COT" sheets)
file_path = "data/data.xlsx"
xls = pd.ExcelFile(file_path)
sheets_to_load = [sheet for sheet in xls.sheet_names if sheet not in ["NMP", "COT"]]

# Reshape data
data_dict = {sheet: xls.parse(sheet) for sheet in sheets_to_load}
long_data = []
for measure, df in data_dict.items():
    df_melted = df.melt(var_name="Condition", value_name="Value")
    df_melted["Measure"] = measure
    long_data.append(df_melted)

# Combine into a single DataFrame
long_df = pd.concat(long_data, ignore_index=True)

# Run ANOVA with Greenhouse-Geisser and Post-Hoc Tests with Bonferroni Correction
anova_results = []
posthoc_results = {}

for measure in long_df["Measure"].unique():
    measure_df = long_df[long_df["Measure"] == measure]

    # One-way ANOVA
    conditions = measure_df["Condition"].unique()
    groups = [measure_df[measure_df["Condition"] == cond]["Value"].dropna() for cond in conditions]
    
    # Check sphericity using Levene's Test (proxy for Greenhouse-Geisser)
    levene_stat, levene_p = stats.levene(*groups)
    sphericity_violation = levene_p < 0.05  # If True, we assume violation

    # Compute One-Way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Add Greenhouse-Geisser correction warning
    gg_correction = "Yes" if sphericity_violation else "No"
    
    anova_results.append({
        "Measure": measure, 
        "F-stat": f_stat, 
        "p-value": p_value, 
        "Greenhouse-Geisser Applied": gg_correction
    })

    # If ANOVA is significant, run post hoc t-tests with Bonferroni correction
    if p_value < 0.05:
        posthoc_results[measure] = []
        comparisons = list(combinations(conditions, 2))
        p_values = []
        
        for cond1, cond2 in comparisons:
            group1 = measure_df[measure_df["Condition"] == cond1]["Value"].dropna()
            group2 = measure_df[measure_df["Condition"] == cond2]["Value"].dropna()
            t_stat, p_ttest = stats.ttest_ind(group1, group2, equal_var=False)
            p_values.append(p_ttest)
        
        # Apply Bonferroni correction
        bonferroni_corrected = np.minimum(np.array(p_values) * len(comparisons), 1.0)  # Max p-value is 1
        
        # Store post-hoc results
        for (cond1, cond2), p_corr, t_stat in zip(comparisons, bonferroni_corrected, p_values):
            posthoc_results[measure].append({
                "Comparison": f"{cond1} vs {cond2}",
                "T-stat": t_stat,
                "p-value (Bonferroni)": p_corr
            })

# Convert ANOVA results to DataFrame
anova_df = pd.DataFrame(anova_results)

# Save ANOVA results to an Excel file
os.makedirs("stats_results", exist_ok=True)
anova_df.to_excel("stats_results/ANOVA_Results.xlsx", index=False)

# Save Post-Hoc results if available
if posthoc_results:
    posthoc_dfs = []
    for measure, comparisons in posthoc_results.items():
        df = pd.DataFrame(comparisons)
        df["Measure"] = measure
        posthoc_dfs.append(df)

    posthoc_df = pd.concat(posthoc_dfs, ignore_index=True)
    posthoc_df.to_excel("stats_results/PostHoc_Results.xlsx", index=False)

# Create folder for figures
figures_folder = "figures"
os.makedirs(figures_folder, exist_ok=True)

 # Generate box-whisker plots
for measure in long_df["Measure"].unique():
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Condition", y="Value", data=long_df[long_df["Measure"] == measure], showfliers=False)
    plt.title(f"Box Plot for {measure}")
    plt.xticks(rotation=45)
    
    # Save figure
    plt.savefig(os.path.join(figures_folder, f"{measure}_boxplot.png"))
    plt.close()

# Print ANOVA results
print("ANOVA Results:\n", anova_df.to_string(index=False))

# Display post hoc test results
if posthoc_results:
    print("\nPost Hoc T-Tests (Bonferroni corrected):")
    for measure, comparisons in posthoc_results.items():
        print(f"\n{measure}:")
        print(pd.DataFrame(comparisons).to_string(index=False))

print(f"\nAll figures have been saved in the '{figures_folder}' folder.")
print(f"\nStatistical results saved in 'stats_results' folder.")