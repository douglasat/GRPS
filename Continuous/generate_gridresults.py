import subprocess
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
import os
import argparse

def plot_sequence(data_dict, topk):
    for k in topk:
        # Extract data into separate lists
        X1 = []
        X2 = []
        X3 = []
        out_x = []
            
        for key, value in data_dict.items():
            if key[2] == k:
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
            
        ax.set_xlabel('Merge', fontsize=12)
        ax.set_ylabel('Prune', fontsize=12)
        ax.set_zlabel('PPV(%)', fontsize=12)
        plt.title("Top-k = %d" % (k))
        
        # Define custom tick values for x and y axes
        tick_values = np.arange(0, 2.2, 0.2)  # Generate values from 0 to 2 with step 0.2
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)

        # Set a fixed view position
        ax.view_init(elev=30, azim=45)  # Adjust these values as needed

    plt.show()

def plot_group(data_dict):
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
    plt.title("Top-k = [1, 5, 10, 15]")

    # Define custom tick values for x and y axes
    tick_values = np.arange(0, 2.2, 0.2)  # Generate values from 0 to 2 with step 0.2
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    # Set a fixed view position
    ax.view_init(elev=30, azim=45)  # Adjust these values as needed
    plt.show()

def get_allscenarios():
    all_scenarios = []
    for k in [1,5,10,15]:
        # Filter out columns where the third element matches the exclude_value
        filtered_columns = [col for col in df.columns if col[2] == k]

        # Create a new DataFrame with the filtered columns
        filtered_df = df[filtered_columns]
        
        prune_merge = np.array(filtered_df.idxmax(axis=1))
        prune_merge = np.array([np.array(p) for p in prune_merge])
        max_scena = filtered_df.max(axis=1)*100

        # max_scena['prune'] = prune_merge[:, 0]
        # max_scena['merge'] = prune_merge[:, 1]
        
        all_scenarios.append(max_scena)
        all_scenarios.append(prune_merge[:, 0])
        all_scenarios.append(prune_merge[:, 1])
    

    all_scenarios = pd.DataFrame(all_scenarios)
    all_scenarios = all_scenarios.T
    all_scenarios.loc[len(all_scenarios)] = all_scenarios.mean()
 
    
    # Add custom row (index) and column labels
    all_scenarios.columns =  ['PPV', 'merge', 'prune', 'PPV', 'merge', 'prune', 'PPV', 'merge', 'prune', 'PPV', 'merge', 'prune'] # Rename columns
    f = files[:10]
    f.insert(len(f), 'Average')
    all_scenarios.index = f  # Rename index

    # Save the DataFrame to a CSV file
    all_scenarios.to_csv("./%s/all_scenarios.csv" % (method), index=True)  # Save index in the CSV file

    print(all_scenarios)
    
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

    all_dicts  = []
    for file in files[:10]:
        with open('./%s/%s_grid_dict.pkl' % (method, file), "rb") as pkl_file:
                data_dict = pickle.load(pkl_file)

        all_dicts.append(data_dict)


    df = pd.DataFrame(all_dicts)
    df = df.dropna(axis=1)

    # Generate grid table for the 10 scenarios
    get_allscenarios()
    
    # Group columns by the first two elements of their names
    grouped_columns = df.groupby(lambda col: (col[0], col[1]), axis=1)
    
    # Calculate the mean for each group
    df_mean = grouped_columns.mean()
    mean_per_group = df_mean.mean(axis=0)

    plot_group(df_mean.mean(axis=0).to_dict())
    
    plot_sequence(df.mean(axis=0).to_dict(), [1, 5, 10, 15])
    