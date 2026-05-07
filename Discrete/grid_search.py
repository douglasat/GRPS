import subprocess
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
import os
import argparse

def gridTreeSearch(problem, topk, method):
    # Load existing dict if it exists
    if os.path.exists(f'./Grid_Data/{method}/{problem}/{problem}_grid_dict.pkl'):
        with open(f'./Grid_Data/{method}/{problem}/{problem}_grid_dict.pkl', "rb") as pkl_file:
            data_dict = pickle.load(pkl_file)
    else:
        data_dict = {}

    prune = np.linspace(0, 2, num=11, endpoint=True)
    merge = np.linspace(0, 2, num=11, endpoint=True)

    for m in merge:
        m = round(m, 1)
        for p in prune:
            p = round(p, 1)
            subprocess.run(f'python3 ./test_domain.py "experiments/" {problem}-optimal "vector_PS" {str(m)} {str(p)} {method} {topk}', shell=True)
      
            scenario_results = pd.read_csv(f'./data-tables/{problem}-optimal-vector_PS.txt', sep="\t")
            data_dict[(m, p, topk)] = scenario_results

            with open(f'./Grid_Data/{method}/{problem}/{problem}_grid_dict.pkl', "wb") as pkl_file:
                pickle.dump(data_dict, pkl_file)
    
    return data_dict

def plot_grid_search(data_dict, method, problem):
    # Extract data into separate lists
    X1 = []
    X2 = []
    out_x = []

    for key, value in data_dict.items():
        X1.append(key[0])
        X2.append(key[1])
        out_x.append(value['PPV'].values[0])
        # if np.isnan(value.values[0]):
        #     print(key)
        #     print(value.values[0])

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
    tick_values = np.arange(0, 2,1)  # Generate values from 0 to 2 with step 0.2
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    # Set a fixed view position
    ax.view_init(elev=30, azim=45)  # Adjust these values as needed
    plt.savefig(f'./Grid_Data/{method}/{problem}/{problem}_grid_search.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def table_generate(data_dict, topk, problem, method):
    # Extract data into separate lists
    X1 = []
    X2 = []
    out_x = []

    for key, value in data_dict.items():
        if key[2] == topk:
            X1.append(key[0])
            X2.append(key[1])
            out_x.append(value['PPV'].values[0])
    
    # Convert lists to numpy arrays
    merge = np.array(X1)
    prune = np.array(X2)
    ppv = np.array(out_x) * 100
    print(ppv)
    bla

    max_ppv_index = np.argmax(ppv)
    data = {
    'PPV(%)': float(ppv[max_ppv_index]),
    'merge': float(merge[max_ppv_index]),
    'prune': float(prune[max_ppv_index])
    }
    
    # Create single-row DataFrame with problem as the index
    new_row_df = pd.DataFrame([data], index=[problem])
    new_row_df.index.name = 'Problem'  # Important for proper CSV format

    # Path to CSV
    csv_path = f'./results/{method}_grid_results_{topk}.csv'

    # Append or update
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='Problem')
        df.loc[problem] = new_row_df.loc[problem]  # Overwrite or insert
    else:
        df = new_row_df

    # Save clean CSV
    df.to_csv(csv_path)
    

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

    # problem = ['blocks-world', 'depots', 'driverlog', 'dwr', 'ferry']
    # method = 'GRPS'
    # best_grps = []
    # best_grps_dtw = []
    # for p in problem:
    #     with open(f'./Grid_Data/{method}/{p}/{p}_grid_dict.pkl', "rb") as pkl_file:
    #         data_dict = pickle.load(pkl_file)
        
    #     best_grps.append(data_dict.get((0.0, 0.0, 5), None)[0])

    
    # method = 'GRPS+DTW'
    # for p in problem:
    #     with open(f'./Grid_Data/{method}/{p}/{p}_grid_dict.pkl', "rb") as pkl_file:
    #         data_dict = pickle.load(pkl_file)
        
    #     best_grps_dtw.append(data_dict.get((0.0, 0.0, 5), None)['PPV'].values[0])
    
    # print(best_grps)
    # print(best_grps_dtw)

    # bla

    
    # topk = 5
    # method = 'GRPS'
    # problem = ['blocks-world', 'depots', 'driverlog', 'dwr', 'ferry']

    # for p in problem:
    #     with open(f'./Grid_Data/{method}/{p}/{p}_grid_dict.pkl', "rb") as pkl_file:
    #         data_dict = pickle.load(pkl_file)
    #     table_generate(data_dict, topk, p, method)
    # bla

    for meth in ['GRPS', 'GRPS+DTW']:
        for topk in [1, 5, 10]:
            for p in problem:
                os.makedirs(f'./Grid_Data/{method}/{p}', exist_ok=True)
                data_dict = gridTreeSearch(p, topk, method)
                table_generate(data_dict, topk, p, method)

    
    #plot_grid_search(data_dict, method, 'ferry')
    