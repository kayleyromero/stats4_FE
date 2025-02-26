import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spm1d
import os

# Load the EMG data (ensure correct formatting)
df_emg = pd.read_excel("data/emgdata.xlsx")

# Define conditions
conditions = ["p20", "p10", "pref", "m10", "m20", "nopert"]

# Define the correct muscle names for left and right
muscles = {
    "BICEPSFEM_LT_uV": "BICEPSFEM_RT_uV",
    "LAT_GASTROLT_uV": "LAT_GASTRORT_uV",
    "MED_GASTROLT_uV": "MED_GASTRORT_uV",
    "SOLEUSLT_uV": "SOLEUSRT_uV",
    "VLOLT_uV": "VLORT_uV",
    "RECTUSFEM_LT_uV": "RECTUSFEM_RT_uV",
    "TIB_ANT_LT_uV": "TIB_ANT_RT_uV",
}

# Create directories for output
os.makedirs("figures/left_vs_right", exist_ok=True)
os.makedirs("stats_results", exist_ok=True)

# Store statistical results
results_table = []

# Loop through conditions and compare left vs. right for each muscle
for condition in conditions:
    for muscle_left, muscle_right in muscles.items():
        # Filter data for this condition and specific muscles
        df_left = df_emg[(df_emg["Condition"] == condition) & (df_emg["Muscle"] == muscle_left)]
        df_right = df_emg[(df_emg["Condition"] == condition) & (df_emg["Muscle"] == muscle_right)]
        
        # Pivot to format: timepoints × subjects
        left_matrix = df_left.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values
        right_matrix = df_right.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values
        
        # Ensure both matrices have the same number of subjects
        if left_matrix.shape != right_matrix.shape:
            print(f"Skipping {condition} - {muscle_left} vs {muscle_right}: Unequal subject counts")
            continue
        
        # Run SPM paired t-test
        ttest = spm1d.stats.ttest_paired(left_matrix.T, right_matrix.T)
        ttest_result = ttest.inference(alpha=0.05)

        # Extract significant gait phases
        significant_time_periods = []
        if ttest_result.h0reject:
            for cluster in ttest_result.clusters:
                try:
                    # Extract start and end time from `endpoints`
                    if hasattr(cluster, "endpoints") and isinstance(cluster.endpoints, tuple):
                        start_time, end_time = cluster.endpoints  # Start and end of significance
                    else:
                        print(f"⚠ Unexpected cluster format in {condition}, {muscle_left} vs {muscle_right}: {cluster}")
                        continue  # Skip invalid clusters
                    
                    # Convert from time-based to gait cycle percentage
                    start = start_time * (100 / left_matrix.shape[0])
                    end = end_time * (100 / left_matrix.shape[0])

                    significant_time_periods.append(f"{start:.1f}% - {end:.1f}%")
                except Exception as e:
                    print(f"❌ Error processing clusters in {condition}, {muscle_left} vs {muscle_right}: {e}")



        # Store results in table
        results_table.append({
            "Condition": condition,
            "Muscle_Left": muscle_left,
            "Muscle_Right": muscle_right,
            "Significant Gait Phases": "; ".join(significant_time_periods) if significant_time_periods else "None"
        })

        # Plot results
        plt.figure(figsize=(8, 4))
        plt.plot(np.linspace(0, 100, left_matrix.shape[0]), ttest_result.z, label="t-statistic")
        plt.axhline(y=ttest_result.zstar, color="r", linestyle="--", label=f"Critical Threshold (t={ttest_result.zstar:.2f})")
        plt.axhline(y=-ttest_result.zstar, color="r", linestyle="--")
        plt.fill_between(np.linspace(0, 100, left_matrix.shape[0]), ttest_result.z, where=ttest_result.z > ttest_result.zstar, color="gray", alpha=0.3)
        plt.fill_between(np.linspace(0, 100, left_matrix.shape[0]), ttest_result.z, where=ttest_result.z < -ttest_result.zstar, color="gray", alpha=0.3)
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("t-statistic")
        plt.title(f"SPM Paired t-Test: {muscle_left} vs {muscle_right}\nCondition: {condition}")
        plt.legend()
        plt.savefig(f"figures/left_vs_right/{condition}_{muscle_left}_vs_{muscle_right}.png", dpi=300)
        plt.close()

# Convert results to DataFrame and save as Excel file
df_results = pd.DataFrame(results_table)
df_results.to_excel("stats_results/SPM_Left_vs_Right_Results.xlsx", index=False)

print("✅ Left vs. Right analysis completed! Figures saved in 'figures/left_vs_right' and results in 'stats_results/SPM_Left_vs_Right_Results.xlsx'.")
