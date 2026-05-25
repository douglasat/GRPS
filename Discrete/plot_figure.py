import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_BarGraph_General():

    # Provided data
    observations = ['30%', '50%', '70%', '100%']

    # Set the width of the bars
    bar_width = 0.15

    # Set the positions of bars on X-axis
    n_bars = 6
    bar_positions = np.arange(len(observations))
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    # Mirroring
    scenario_results = pd.read_csv('./results/mirroring.csv')
    average_mirroring = [scenario_results[f'ppv_{a}'].mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_mirroring = [100*1.96*scenario_results[f'ppv_{a}'].std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # Landmarks
    scenario_results = pd.read_csv('./results/landmarks.csv')
    average_landmarks = [scenario_results[f'ppv_{a}'].mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_landmarks = [100*1.96*scenario_results[f'ppv_{a}'].std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # Mirroring with Landmarks
    scenario_results = pd.read_csv('./results/mirroring_landmarks.csv')
    average_mirroring_landmarks = [scenario_results[f'ppv_{a}'].mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_mirroring_landmarks = [100*1.96*scenario_results[f'ppv_{a}'].std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # Vector
    scenario_results = pd.read_csv('./results/vector_inference.csv')
    average_vector = [scenario_results[f'ppv_{a}'].mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_vector = [100*1.96*scenario_results[f'ppv_{a}'].std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # GRPS
    scenario_results = pd.read_csv('./results/GRPS_15.csv')
    scenario_results = scenario_results[0:12]
    average_grps = [scenario_results[f'ppv_{a}'].astype(float).mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_grps = [100*1.96*scenario_results[f'ppv_{a}'].astype(float).std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # GRPS+DTW
    scenario_results = pd.read_csv('./results/GRPS+DTW_15.csv')
    scenario_results = scenario_results[0:12]
    average_grps_dtw = [scenario_results[f'ppv_{a}'].astype(float).mean() * 100 for a in [30, 50, 60, 100]]
    margin_error_grps_dtw = [100*1.96*scenario_results[f'ppv_{a}'].astype(float).std() / np.sqrt(len(scenario_results)) for a in [30, 50, 60, 100]] 

    # Plotting
    fig, ax = plt.subplots(figsize=(17.6, 6), dpi=300)

    plt.bar(bar_positions + offsets[0], average_mirroring, width=bar_width, label='Mirroring', align='center', yerr=margin_error_mirroring, capsize=4, color='blue', hatch="//")
    plt.bar(bar_positions + offsets[1], average_landmarks, width=bar_width, label='Landmarks', align='center', yerr=margin_error_landmarks, capsize=4, color='gray', hatch="\\")
    plt.bar(bar_positions + offsets[2], average_mirroring_landmarks, width=bar_width, label='Mirroring with Landmarks', align='center', yerr=margin_error_mirroring_landmarks, capsize=4, color='pink', hatch="o")
    plt.bar(bar_positions + offsets[3], average_vector, width=bar_width, label='Vector', align='center', yerr=margin_error_vector, capsize=4, color='red', hatch="||")
    plt.bar(bar_positions + offsets[4], average_grps, width=bar_width, label='GRPS', align='center', yerr=margin_error_grps, capsize=4, color='green', hatch='xx')
    plt.bar(bar_positions + offsets[5], average_grps_dtw, width=bar_width, label='GRPS+DTW', align='center', yerr=margin_error_grps_dtw, capsize=4, color='purple', hatch='-')

    # Adding labels and title
    ax.set_title('(b) Discrete Domains', fontsize=28)
    ax.set_xlabel('Percentage of Observation', fontsize=28)
    ax.set_ylabel('PPV(%)', fontsize=28)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(observations, fontsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.legend(fontsize=28, loc='upper center', ncol=3, borderpad=0.2, columnspacing=1.5)
    ax.set_ylim(0, 85)

    # Set font size of axes labels
    # fig.gca().xaxis.label.set_fontsize(28)
    # fig.gca().yaxis.label.set_fontsize(28)

    fig.tight_layout()
    fig.savefig('./Figures/discrete_comp.pdf')  # Save
    fig.clf()  # Clear the current figure

if __name__ == "__main__":

    plot_BarGraph_General()


    
   
            