import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the RMS EMG data
df_emg = pd.read_excel("data/emgdata.xlsx")

# Load the significant results table
df_results = pd.read_excel("stats_results/SPM_Pairwise_Results_All.xlsx")

# Define conditions and legs
conditions = ["p20", "p10", "pref", "m10", "m20", "nopert"]
legs = ["Left", "Right"]

# Define colors for muscles
muscles_left = ['BICEPSFEM_LT_uV', 'LAT_GASTROLT_uV', 'MED_GASTROLT_uV', 'SOLEUSLT_uV', 'VLOLT_uV', 'RECTUSFEM_LT_uV', 'TIB_ANT_LT_uV']
muscles_right = ['BICEPSFEM_RT_uV', 'LAT_GASTRORT_uV', 'MED_GASTRORT_uV', 'SOLEUSRT_uV', 'VLORT_uV', 'RECTUSFEM_RT_uV', 'TIB_ANT_RT_uV']
colors = plt.cm.viridis(np.linspace(0, 1, len(muscles_left)))

# Function to create subplots for a specific leg
def plot_leg_comparison(leg, save_path):
    muscles = muscles_left if leg == "Left" else muscles_right

    # Reduce figure size slightly to prevent overlap
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    all_handles = []
    all_labels = []

    # Loop through each condition
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        df_emg_filtered = df_emg[(df_emg["Condition"] == condition) & (df_emg["Leg"] == leg)]
        df_results_filtered = df_results[(df_results["Condition"] == condition) & (df_results["Leg"] == leg)]
        n_timepoints = df_emg_filtered["TimePoint"].max()

        # Loop through muscles and plot RMS signal
        for i, muscle in enumerate(muscles):
            muscle_data = df_emg_filtered[df_emg_filtered["Muscle"] == muscle]
            emg_matrix = muscle_data.pivot(index="TimePoint", columns="Subject", values="EMG_Activity").values
            mean_rms = np.mean(np.sqrt(np.square(emg_matrix)), axis=1)

            line, = ax.plot(np.linspace(0, 100, n_timepoints), mean_rms, label=muscle, color=colors[i])

            if muscle not in all_labels:
                all_handles.append(line)
                all_labels.append(muscle)

        # Overlay significance
        for _, row in df_results_filtered.iterrows():
            sig_phases = str(row["Significant Gait Phases"])
            if sig_phases != "None" and sig_phases.lower() != "nan":
                significant_blocks = sig_phases.split("; ")
                for block in significant_blocks:
                    try:
                        start, end = map(float, block.replace("%", "").split("-"))
                        ax.axvspan(start, end, color="gray", alpha=0.3)
                    except ValueError:
                        print(f"Warning: Skipping invalid phase format: {block}")

        # Formatting
        ax.set_title(f"{condition}", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 70)
        ax.grid(True)

        if idx >= 3:
            ax.set_xlabel("Gait Cycle (%)", fontsize=12)
        if idx % 3 == 0:
            ax.set_ylabel("RMS EMG Activity (µV)", fontsize=12)

    # Adjust layout to fit everything nicely
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.12, wspace=0.3, hspace=0.4)

    # Move legend higher so it does not overlap with titles
    fig.legend(all_handles, all_labels, loc="upper center", fontsize=10, title="Muscles", ncol=4, bbox_to_anchor=(0.5, 1))

    # Save and show figure
    plt.savefig(save_path, dpi=300)
    plt.show()

# Generate and save figures for Left and Right legs
plot_leg_comparison("Left", "figures/RMS_EMG_Comparison_Left_Leg.png")
plot_leg_comparison("Right", "figures/RMS_EMG_Comparison_Right_Leg.png")