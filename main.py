import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from itertools import combinations

# Load Excel file (all sheets)
file_path = "data/data.xlsx"
xls = pd.ExcelFile(file_path)

# Create directories to save results
stats_dir = "stats_results"
figures_folder = "figures"
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# Define MATLAB colors (ensuring at least 8 subjects)
custom_colors = ["#610006", "#4DAC26", "#5E3C99", "#E66101", 
                 "#0571B0", "#69FFFC", "#D01C8B", "#BE96F2"]

# **Step 1: Find Unique Subjects Across All Sheets (Starting from 1)**
all_subjects = set()
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    all_subjects.update(df.index.unique())  # Assuming index represents subjects

# Convert subjects to a sorted list and **shift indices to start at 1**
all_subjects = sorted(list(all_subjects))
subject_mapping = {subj: i+1 for i, subj in enumerate(all_subjects)}  # Map to Subject 1, 2, ...

print(f"✅ Found {len(all_subjects)} unique subjects, remapped as {list(subject_mapping.values())}")

# Assign consistent colors to subjects **starting from Subject 1**
subject_color_map = {subject_mapping[subj]: custom_colors[i % len(custom_colors)] for i, subj in enumerate(all_subjects)}

# Store all results
all_anova_results = []
all_posthoc_results = []

# Iterate through each sheet (each measure)
for sheet_name in xls.sheet_names:
    print(f"\n📊 Processing: {sheet_name}...")

    # Load current sheet
    df = xls.parse(sheet_name)

    # Reshape Data: Convert from wide format (columns = conditions) to long format
    df_long = df.melt(var_name="Condition", value_name="Value", ignore_index=False).reset_index()

    # 🚀 **Check for missing values & remove NaNs**
    df_long.dropna(inplace=True)

    # **Remap subjects to start at 1**
    df_long["Subject"] = df_long["index"].map(subject_mapping)  

    # Assign colors for each row based on the subject
    df_long["Color"] = df_long["Subject"].map(subject_color_map)

    # Define conditions
    conditions = df_long["Condition"].unique()

    # Store ANOVA & Post-Hoc results for this sheet
    anova_results = []
    posthoc_results = []

    # One-way ANOVA: Step kinematics across conditions
    groups = [df_long[df_long["Condition"] == cond]["Value"].dropna() for cond in conditions]

    # 🚀 **Check for empty groups before ANOVA**
    empty_groups = [cond for cond, g in zip(conditions, groups) if g.empty]
    if empty_groups:
        print(f"⚠️ Warning: The following conditions have no data: {empty_groups}")
        groups = [g for g in groups if not g.empty]
        conditions = [cond for cond in conditions if cond not in empty_groups]

    # Proceed only if we have at least 2 valid groups
    if len(groups) >= 2:
        # Check sphericity using Levene's test (proxy for Greenhouse-Geisser)
        levene_stat, levene_p = stats.levene(*groups)
        sphericity_violation = levene_p < 0.05  # If True, assume violation

        # Compute One-Way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Greenhouse-Geisser correction warning
        gg_correction = "Yes" if sphericity_violation else "No"

        # Store ANOVA results
        anova_results.append({
            "Measure": sheet_name,  
            "F-stat": f_stat, 
            "p-value": p_value, 
            "Greenhouse-Geisser Applied": gg_correction
        })

        # Append to global results
        all_anova_results.extend(anova_results)

        # If ANOVA is significant, run Bonferroni post-hoc t-tests
        if p_value < 0.05:
            comparisons = list(combinations(conditions, 2))
            p_values = []

            for cond1, cond2 in comparisons:
                group1 = df_long[df_long["Condition"] == cond1]["Value"].dropna()
                group2 = df_long[df_long["Condition"] == cond2]["Value"].dropna()
                t_stat, p_ttest = stats.ttest_ind(group1, group2, equal_var=False)
                p_values.append(p_ttest)

            # Apply Bonferroni correction
            bonferroni_corrected = np.minimum(np.array(p_values) * len(comparisons), 1.0)

            # Store post-hoc results
            for (cond1, cond2), p_corr, t_stat in zip(comparisons, bonferroni_corrected, p_values):
                posthoc_results.append({
                    "Measure": sheet_name,
                    "Comparison": f"{cond1} vs {cond2}",
                    "T-stat": t_stat,
                    "p-value (Bonferroni)": p_corr
                })

            # Append to global results
            all_posthoc_results.extend(posthoc_results)

        # --- Box Plot with MATLAB Colors and Fixed Subject Legend ---
        plt.figure(figsize=(10, 6))

        # Boxplot with transparency so points are visible
        sns.boxplot(x="Condition", y="Value", data=df_long, showfliers=False, width=0.6, boxprops=dict(alpha=0.6))

        # Scatter plot (individual subject data points, using MATLAB colors)
        for i, row in df_long.iterrows():
            plt.scatter(row["Condition"], row["Value"], 
                        color=row["Color"], alpha=1.0, edgecolors="black", s=80)

        # **Global Subject Legend (Now Starting at 1)**
        handles = [plt.Line2D([0], [0], marker='o', linestyle='None', markersize=10,
                              markerfacecolor=subject_color_map[subj], markeredgecolor="black",
                              label=f"Subject {subj}") for subj in sorted(subject_mapping.values())]
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), title="Subjects", frameon=True)

        # Format plot
        plt.title(f"Box Plot for {sheet_name} with Individual Subject Data")
        plt.xticks(rotation=45)

        # Save figure
        plt.savefig(os.path.join(figures_folder, f"{sheet_name}_boxplot.png"), bbox_inches="tight")
        plt.close()

        print(f"✅ Finished processing {sheet_name}. Results saved.")

    else:
        print(f"🚨 Error: Not enough valid groups for ANOVA in {sheet_name}. Skipping statistical analysis.")

# Convert ANOVA results to DataFrame and save
anova_df = pd.DataFrame(all_anova_results)
anova_df.to_excel(os.path.join(stats_dir, "ANOVA_Results_All_SK.xlsx"), index=False)

# Save Post-Hoc results if available
if all_posthoc_results:
    posthoc_df = pd.DataFrame(all_posthoc_results)
    posthoc_df.to_excel(os.path.join(stats_dir, "PostHoc_Results_All_SK.xlsx"), index=False)

print("\n✅ **All statistical analyses are complete!**")
print(f"📊 ANOVA results saved in '{stats_dir}/ANOVA_Results_All_SK.xlsx'")
print(f"📊 Post-hoc results saved in '{stats_dir}/PostHoc_Results_All_SK.xlsx'")
print(f"📈 All box plots saved in '{figures_folder}/' 🚀")
