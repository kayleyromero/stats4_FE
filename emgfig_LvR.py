import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load EMG data and statistical results
df_emg = pd.read_excel("data/emgdata.xlsx")
df_results = pd.read_excel("stats_results/SPM_Left_vs_Right_Results.xlsx")

# Define conditions
conditions = ["p20", "p10", "pref", "m10", "m20", "nopert"]

# Define muscles (Left ↔ Right pairing)
muscles = {
    "BICEPSFEM_LT_uV": "BICEPSFEM_RT_uV",
    "LAT_GASTROLT_uV": "LAT_GASTRORT_uV",
    "MED_GASTROLT_uV": "MED_GASTRORT_uV",
    "SOLEUSLT_uV": "SOLEUSRT_uV",
    "VLOLT_uV": "VLORT_uV",
    "RECTUSFEM_LT_uV": "RECTUSFEM_RT_uV",
    "TIB_ANT_LT_uV": "TIB_ANT_RT_uV",
}

# Create output folder if it doesn't exist
os.makedirs("figures/left_vs_right", exist_ok=True)

# Loop through each condition and generate plots
for condition in conditions:
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (muscle_left, muscle_right) in enumerate(muscles.items()):
        ax = axes[idx]

        # Filter EMG data for this condition and muscle
        df_left = df_emg[(df_emg["Condition"] == condition) & (df_emg["Muscle"] == muscle_left)]
        df_right = df_emg[(df_emg["Condition"] == condition) & (df_emg["Muscle"] == muscle_right)]
        
        # Pivot to format: timepoints × subjects
        left_matrix = df_left.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values
        right_matrix = df_right.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values

        # Compute mean activation across subjects
        time_axis = np.linspace(0, 100, left_matrix.shape[0])
        left_mean = np.mean(left_matrix, axis=1)
        right_mean = np.mean(right_matrix, axis=1)

        # Plot Left vs. Right
        ax.plot(time_axis, left_mean, label="Left Leg", color="blue")
        ax.plot(time_axis, right_mean, label="Right Leg", color="red")

        # Find statistical significance results for this muscle & condition
        result_row = df_results[(df_results["Condition"] == condition) &
                                (df_results["Muscle_Left"] == muscle_left) &
                                (df_results["Muscle_Right"] == muscle_right)]
        
        # If there are significant gait phases, highlight them
        if not result_row.empty:
            sig_phases = result_row.iloc[0]["Significant Gait Phases"]
            if sig_phases != "None":
                for phase in sig_phases.split("; "):
                    try:
                        start, end = map(float, phase.replace("%", "").split("-"))
                        ax.axvspan(start, end, color="gray", alpha=0.3)
                    except ValueError:
                        print(f"Warning: Skipping invalid phase format: {phase}")

        # Formatting
        ax.set_title(muscle_left.replace("_uV", ""), fontsize=10)
        ax.set_xlim(0, 100)
        ax.grid(True)

        # Set labels
        if idx >= 6:
            ax.set_xlabel("Gait Cycle (%)")
        if idx % 3 == 0:
            ax.set_ylabel("EMG Activity (µV)")

    # Adjust layout and add legend
    fig.suptitle(f"Left vs. Right Leg Muscle Activation - {condition.upper()}", fontsize=14)
    fig.legend(["Left Leg", "Right Leg"], loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Save figure
    plt.savefig(f"figures/left_vs_right/{condition}_Left_vs_Right_Comparison.png", dpi=300)
    plt.close()

print("✅ Figures saved in 'figures/left_vs_right/' for all conditions.")
