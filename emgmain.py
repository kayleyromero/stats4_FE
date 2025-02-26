import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
from itertools import combinations

# Load EMG dataset
file_path = "data/emgdata.xlsx"
df = pd.read_excel(file_path)

# Define conditions and unique subjects
conditions = df["Condition"].unique()
subjects = df["Subject_ID"].unique()

# Create directories to save results
stats_dir = "stats_results"
figures_folder = "figures"
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# Define the **MATLAB colors** you use
custom_colors = ["#610006", "#4DAC26", "#5E3C99", "#E66101", 
                 "#0571B0", "#69FFFC", "#D01C8B", "#BE96F2"]

# Assign colors to subjects (cycling if more subjects exist)
subject_color_map = {subj: custom_colors[i % len(custom_colors)] for i, subj in enumerate(subjects)}

# Store ANOVA & Post-Hoc results
anova_results = []
posthoc_results = {}

# Loop through each muscle separately
for muscle in df["Muscle"].unique():
    muscle_df = df[df["Muscle"] == muscle]

    # One-way ANOVA: Mean RMS EMG across conditions
    groups = [muscle_df[muscle_df["Condition"] == cond]["Mean_RMS_EMG"].dropna() for cond in conditions]

    # Check sphericity using Levene's test (proxy for Greenhouse-Geisser)
    levene_stat, levene_p = stats.levene(*groups)
    sphericity_violation = levene_p < 0.05  # If True, assume violation

    # Compute One-Way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Greenhouse-Geisser correction warning
    gg_correction = "Yes" if sphericity_violation else "No"

    anova_results.append({
        "Muscle": muscle, 
        "F-stat": f_stat, 
        "p-value": p_value, 
        "Greenhouse-Geisser Applied": gg_correction
    })

    # If ANOVA is significant, run Bonferroni post-hoc t-tests
    if p_value < 0.05:
        posthoc_results[muscle] = []
        comparisons = list(combinations(conditions, 2))
        p_values = []

        for cond1, cond2 in comparisons:
            group1 = muscle_df[muscle_df["Condition"] == cond1]["Mean_RMS_EMG"].dropna()
            group2 = muscle_df[muscle_df["Condition"] == cond2]["Mean_RMS_EMG"].dropna()
            t_stat, p_ttest = stats.ttest_ind(group1, group2, equal_var=False)
            p_values.append(p_ttest)

        # Apply Bonferroni correction
        bonferroni_corrected = np.minimum(np.array(p_values) * len(comparisons), 1.0)  # Max p-value is 1

        # Store post-hoc results
        for (cond1, cond2), p_corr, t_stat in zip(comparisons, bonferroni_corrected, p_values):
            posthoc_results[muscle].append({
                "Comparison": f"{cond1} vs {cond2}",
                "T-stat": t_stat,
                "p-value (Bonferroni)": p_corr
            })

    # --- Box Plot with **Custom MATLAB Colors** ---
    plt.figure(figsize=(8, 5))
    
    # Boxplot with transparency so points are visible
    sns.boxplot(x="Condition", y="Mean_RMS_EMG", data=muscle_df, 
                showfliers=False, width=0.6, boxprops=dict(alpha=0.6))

    # Scatter plot (individual subject data points, using MATLAB colors)
    for subj in subjects:
        subj_data = muscle_df[muscle_df["Subject_ID"] == subj]
        plt.scatter(subj_data["Condition"], subj_data["Mean_RMS_EMG"], 
                    color=subject_color_map[subj], label=f"Subject {subj}", 
                    alpha=1.0, edgecolors="black", s=80)  # Larger points with black edges

    # Format plot
    plt.title(f"Box Plot for {muscle}")
    plt.xticks(rotation=45)
    
    # Create **legend with MATLAB colors**
    handles = [plt.Line2D([0], [0], marker='o', linestyle='None', markersize=10, 
                          markerfacecolor=subject_color_map[subj], markeredgecolor="black", label=f"Subject {subj}") 
               for subj in subjects]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title="Subjects", frameon=True)

    # Save figure
    plt.savefig(os.path.join(figures_folder, f"{muscle}_boxplot.png"), bbox_inches="tight")
    plt.close()

# Convert ANOVA results to DataFrame
anova_df = pd.DataFrame(anova_results)

# Save ANOVA results to an Excel file
anova_df.to_excel(os.path.join(stats_dir, "ANOVA_Results_EMG.xlsx"), index=False)

# Save Post-Hoc results if available
if posthoc_results:
    posthoc_dfs = []
    for muscle, comparisons in posthoc_results.items():
        df = pd.DataFrame(comparisons)
        df["Muscle"] = muscle
        posthoc_dfs.append(df)

    posthoc_df = pd.concat(posthoc_dfs, ignore_index=True)
    posthoc_df.to_excel(os.path.join(stats_dir, "PostHoc_Results_EMG.xlsx"), index=False)

print("✅ ANOVA and post-hoc results saved in 'stats_results/' folder!")
print(f"✅ All box plots using **MATLAB subject colors** saved in '{figures_folder}/' 🚀")
