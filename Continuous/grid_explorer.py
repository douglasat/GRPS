import subprocess
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
import os
import argparse

def gridTreeSearch(data_dict, file):
    if method == 'PS_grid_data':
        m = 'grps'
    if method == 'DTW_grid_data':
        m = 'grps_dtw'

    prune = np.linspace(0, 2, num=11, endpoint=True)
    merge = np.linspace(0, 2, num=11, endpoint=True)
    topk = [1,5,10,15]

    for k in topk:
        for m in merge:
            m = round(m, 1)
            for p in prune:
                p = round(p, 1)
                parallel_var = 10
                if parallel_var < 1:
                    parallel_var = 1
                    
                if data_dict.get((m, p, k), None) == None or np.isnan(data_dict.get((m, p, k))):
                    subprocess.run('python3 %s.py -map %s -par %d -m %.1f -p %.1f -k %d' % (m, file, parallel_var, m, p, k), shell=True)
                    scenario_results = pd.read_csv('./%s/%s_%s_grid.csv' % (file, file, m))
                    data_dict[(m, p, k)] = scenario_results['PPV_total'].mean()

                    with open("./%s/%s_grid_dict.pkl" % (method, file), "wb") as pkl_file:
                        pickle.dump(data_dict, pkl_file)
    
    return data_dict


    # Extract data into separate lists
    X1 = []
    X2 = []
    out_x = []
            
    for key, value in data_dict.items():
        X1.append(key[0])
        X2.append(key[1])
        out_x.append(value)
        if np.isnan(value):
            print(key)
            print(value)

    # Convert lists to numpy arrays
    X1 = np.array(X1)
    X2 = np.array(X2)
    out = np.array(out_x) * 100

    # Create unique grid points
    unique_x1 = np.unique(X1)
    unique_x2 = np.unique(X2)
    X1_grid, X2_grid = np.meshgrid(unique_x1, unique_x2)

    # Map 'out' values to the grid
    out_grid = np.full_like(X1_grid, np.nan, dtype=float)  # Initialize grid with NaNs
    for i in range(len(X1)):
        # Find the indices of the current X1, X2 in the grid
        x1_idx = np.where(unique_x1 == X1[i])[0][0]
        x2_idx = np.where(unique_x2 == X2[i])[0][0]
        out_grid[x2_idx, x1_idx] = out[i]  # Assign the 'out' value

    # Check for unfilled grid points
    if np.isnan(out_grid).any():
        print("Warning: Some grid points are missing data!")

    # Create a 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
            
    # Define the min and max values for color scaling
    vmin = np.min(out_grid)  # Set minimum value for color bar
    vmax = np.max(out_grid)  # Set maximum value for color bar

    # Use a logarithmic color scale
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot the surface
    surf = ax.plot_surface(X1_grid, X2_grid, out_grid, cmap='viridis', alpha=1)

    # Add a color bar and set the limits for color bar scale
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # cbar.set_label('PPV(%)')
    # cbar.set_ticks([round(a, 1) for a in np.linspace(vmin, vmax, num=5, endpoint=True)])  # Show only the min and max values on the color bar


    # Plot the original points as a scatter plot
    scatter = ax.scatter(X1, X2, out, color='red', label='Original Points')
            
    ax.set_xlabel('Merge')
    ax.set_ylabel('Prune')
    ax.set_zlabel('PPV(%)')
    # ax.tick_params(labelsize=12)
    # plt.title("Top-k = [1, 5, 10]")

    # Define custom tick values for x and y axes
    tick_values = np.arange(0, 2.2, 0.2)  # Generate values from 0 to 2 with step 0.2
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    # Set a fixed view position
    ax.view_init(elev=30, azim=45)  # Adjust these values as needed
    plt.show()

def process_input():
    """Main function to handle inputs and call processing logic."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument(
        "-method", "--method", type=str, required=True, help="The first parameter (str)."
    )
    
    args = parser.parse_args()
    if args.method == 'ps':
        method = 'PS_grid_data'
    if args.method == 'dtw':
        method = 'DTW_grid_data'

    return method

if __name__ == "__main__":
    method = process_input()

    if not os.path.exists('./%s' % method):
        os.makedirs('./%s' % method)

    # Get all files in the directory
    directory_path = os.path.dirname(__file__)
    files = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    files = sorted(files)

    for file in files[:10]:
        if not os.path.exists('./%s/%s_grid_dict.pkl' % (method, file)):
            data_dict = {}
        else:
            with open('./%s/%s_grid_dict.pkl' % (method, file), "rb") as pkl_file:
                data_dict = pickle.load(pkl_file)

        data_dict = gridTreeSearch(data_dict, file)

    