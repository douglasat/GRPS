import subprocess
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
import os
import argparse


def run_experiment(problem, method, merge, prune, topk):
    os.makedirs(f'./Grid_Data/{method}/{problem}', exist_ok=True)
    if method == 'GRPS' or method == 'GRPS+DTW':
        subprocess.run(f'python ./test_domain.py "experiments/" {problem}-optimal "vector_PS" {str(merge)} {str(prune)} {method} {topk}', shell=True)
        df = pd.read_csv(f'./data-tables/{problem}-optimal-vector_PS.txt', sep="\t")
        print(df)
    else:
        subprocess.run(f'python ./test_domain.py "experiments/" {problem}-optimal "{method}" {str(merge)} {str(prune)} {method} {topk}', shell=True)
        df = pd.read_csv(f'./data-tables/{problem}-optimal-{method}.txt', sep="\t")
        print(df)

    new_df = pd.DataFrame({
        "ppv_30": [df.loc[0, "PPV"]],
        "ppv_50": [df.loc[1, "PPV"]],
        "ppv_60": [df.loc[2, "PPV"]],
        "ppv_100": [df.loc[3, "PPV"]],

        "Acc_1": [df.loc[0, "Acc"]],
        "Acc_2": [df.loc[1, "Acc"]],
        "Acc_3": [df.loc[2, "Acc"]],
        "Acc_4": [df.loc[3, "Acc"]],

        "STD":    [df.loc[df.index[-1], "STD"]],
        "Spread": [df.loc[df.index[-1], "Spread"]],
        "PC":     [df.loc[df.index[-1], "PC"]],
        "Offline":[df.loc[df.index[-1], "Offline"]],
        "Online": [df.loc[df.index[-1], "Online"]],
    })

    return new_df


def process_input():
    """Main function to handle inputs and call processing logic."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument(
        "-method", "--method", type=str, required=True, help="The first parameter (str)."
    )
    parser.add_argument(
        "-m", "--merge", type=float, required=True, help="The first parameter (float)."
    )
    parser.add_argument(
        "-p", "--prune", type=float, required=True, help="The second parameter (float)."
    )
    parser.add_argument(
        "-k", "--topk", type=int, required=True, help="The second parameter (float)."
    )
    args = parser.parse_args()
    return args.method, args.merge, args.prune, args.topk

if __name__ == "__main__":

    method, merge, prune, topk = process_input()
    
    problems = ['blocks-world', 'depots', 'driverlog', 'dwr', 'easy-ipc-grid', 'ferry', 'logistics', 'miconic', 'rovers', 'satellite', 'small-sokoban', 'zeno-travel']
    os.makedirs(f'./results', exist_ok=True)

    rows = []
    for problem in problems:
        res = run_experiment(problem, method, merge, prune, topk)   # returns a DataFrame
        last_row = res.iloc[-1].copy()                 # get the last row (as a Series)
        last_row['problem'] = problem           # add the problem name as a column
        rows.append(last_row)                   # collect the row

    # Combine all into a DataFrame
    df = pd.DataFrame(rows)
    df = df.round(3)
    cols = ['problem'] + [col for col in df.columns if col != 'problem']
    df = df[cols]

    if method == 'GRPS' or method == 'GRPS+DTW':   
        df.to_csv(f"./results/{method}_{topk}.csv", index=False)
    else:
        df.to_csv(f"./results/{method}.csv", index=False)

