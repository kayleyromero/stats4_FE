import pandas as pd
import numpy as np
import spm1d
import matplotlib.pyplot as plt
import os
from itertools import combinations  # Generate muscle pairs

# Load dataset
df = pd.read_excel("data/emgdata.xlsx")

# Define conditions and muscles
conditions = df["Condition"].unique()  # Extract unique conditions
muscles_left = ['BICEPSFEM_LT_uV', 'LAT_GASTROLT_uV', 'MED_GASTROLT_uV', 'SOLEUSLT_uV', 'VLOLT_uV', 'RECTUSFEM_LT_uV', 'TIB_ANT_LT_uV']
muscles_right = ['BICEPSFEM_RT_uV', 'LAT_GASTRORT_uV', 'MED_GASTRORT_uV', 'SOLEUSRT_uV', 'VLORT_uV', 'RECTUSFEM_RT_uV', 'TIB_ANT_RT_uV']
legs = ["Left", "Right"]

# Create directories to save results
figure_dir = "figures"
stats_dir = "stats_results"
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)

# Store results
all_results = []

# Loop through each condition and leg
for condition in conditions:
    for leg in legs:
        print(f"\nRunning SPM ANOVA for Condition: {condition}, Leg: {leg}")

        muscles = muscles_left if leg == "Left" else muscles_right
        df_filtered = df[(df["Condition"] == condition) & (df["Leg"] == leg)]

        subjects = df_filtered["Subject"].unique()
        n_subjects = len(subjects)
        n_muscles = len(muscles)
        n_timepoints = df_filtered["TimePoint"].max()

        A_list, SUBJ_list, FACTOR_list = [], [], []

        for subject in subjects:
            for i, muscle in enumerate(muscles):
                emg_values = df_filtered[(df_filtered["Muscle"] == muscle) & (df_filtered["Subject"] == subject)]["EMG_Activity"].values
                if len(emg_values) == n_timepoints:
                    A_list.append(emg_values)
                    SUBJ_list.append(subject)
                    FACTOR_list.append(i)

        A = np.vstack(A_list)
        SUBJ = np.array(SUBJ_list)
        FACTOR = np.array(FACTOR_list)

        # Run SPM ANOVA
        anova = spm1d.stats.anova1rm(A, FACTOR, SUBJ, equal_var=True)
        anova_results = anova.inference(alpha=0.05)

        # Save ANOVA figure
        plt.figure(figsize=(10, 5))
        anova_results.plot()
        plt.axhline(y=anova_results.zstar, color='r', linestyle='--', label=f"Critical Threshold (F={anova_results.zstar:.2f})")
        plt.title(f"SPM ANOVA for {condition} ({leg} Leg)")
        plt.xlabel("Time (% Gait Cycle)")
        plt.ylabel("F-statistic (Muscle Differences)")
        plt.legend()
        anova_fig_path = os.path.join(figure_dir, f"SPM_ANOVA_{condition}_{leg}.png")
        plt.savefig(anova_fig_path, dpi=300)
        plt.close()

        if np.any(anova_results.z > anova_results.zstar):
            print(f"  🔴 ANOVA is significant for {condition} ({leg} leg)! Running pairwise t-tests...")

            muscle_pairs = list(combinations(muscles, 2))
            pairwise_results = []

            for muscle1, muscle2 in muscle_pairs:
                print(f"  Comparing {muscle1} vs. {muscle2}")

                df_m1 = df_filtered[df_filtered["Muscle"] == muscle1].sort_values(["Subject", "TimePoint"])
                df_m2 = df_filtered[df_filtered["Muscle"] == muscle2].sort_values(["Subject", "TimePoint"])

                df_m1 = df_m1.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values
                df_m2 = df_m2.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values

                t_test = spm1d.stats.ttest_paired(df_m1.T, df_m2.T)
                t_results = t_test.inference(alpha=0.05)

                # Detect significant segments separately
                sig_indices = np.where(t_results.z > t_results.zstar)[0]
                sig_blocks = []
                if len(sig_indices) > 0:
                    start = sig_indices[0]
                    for i in range(1, len(sig_indices)):
                        if sig_indices[i] != sig_indices[i - 1] + 1:  
                            sig_blocks.append(f"{start / n_timepoints * 100:.1f}% - {sig_indices[i - 1] / n_timepoints * 100:.1f}%")
                            start = sig_indices[i]
                    sig_blocks.append(f"{start / n_timepoints * 100:.1f}% - {sig_indices[-1] / n_timepoints * 100:.1f}%")
                sig_phase = "; ".join(sig_blocks) if sig_blocks else "None"

                # Store results
                pairwise_results.append([condition, leg, muscle1, muscle2, sig_phase])

                # Save t-test figure
                plt.figure(figsize=(10, 5))
                t_results.plot()
                plt.title(f"SPM Paired t-Test: {muscle1} vs {muscle2}\nCondition: {condition}, Leg: {leg}")
                plt.xlabel("Time (% Gait Cycle)")
                plt.ylabel("t-statistic")
                ttest_fig_path = os.path.join(figure_dir, f"SPM_ttest_{condition}_{leg}_{muscle1}_vs_{muscle2}.png")
                plt.savefig(ttest_fig_path, dpi=300)
                plt.close()

            # Convert to DataFrame
            df_results = pd.DataFrame(pairwise_results, columns=["Condition", "Leg", "Muscle 1", "Muscle 2", "Significant Gait Phases"])
            all_results.append(df_results)

            # Save to CSV & Excel
            csv_filename = os.path.join(stats_dir, f"SPM_Pairwise_Results_{condition}_{leg}.csv")
            excel_filename = os.path.join(stats_dir, f"SPM_Pairwise_Results_{condition}_{leg}.xlsx")
            df_results.to_csv(csv_filename, index=False)
            df_results.to_excel(excel_filename, index=False)
            print(f"📄 Results saved to {csv_filename} & {excel_filename}")

# Combine all results into one table and save
final_results = pd.concat(all_results, ignore_index=True)
final_csv = os.path.join(stats_dir, "SPM_Pairwise_Results_All.csv")
final_excel = os.path.join(stats_dir, "SPM_Pairwise_Results_All.xlsx")
final_results.to_csv(final_csv, index=False)
final_results.to_excel(final_excel, index=False)
print("✅ All results saved in stats_results/ as CSV & Excel files!")