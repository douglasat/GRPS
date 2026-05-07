import os
import pandas as pd
import generate_scenario as gc
import argparse

def process_input():
    """Main function to handle inputs and call processing logic."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument(
        "-method", "--method", type=str, required=True, help="The first parameter (str)."
    )
    args = parser.parse_args()
    return args.method

if __name__ == "__main__":
    method = process_input()

    directory_path = os.path.dirname(__file__)

    # Get all files in the directory
    files = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    files = sorted(files)
    
    header = ['Scenario', 'PPV_1', 'PPV_2', 'PPV_3', 'PPV_4', 'PPV_5', 'PPV_6', 'PPV_total', 'ACC_total', 'spread', 'planner_calls', 'online_time', 'offline_time']
    csv_file_estimation_path_signature = pd.DataFrame(columns=header)
    for file in files[:10]:
        if os.path.exists(directory_path + '/%s' % file):
            if os.path.exists(directory_path + '/%s/%s_%s.csv' % (file, file, method)):
                scenario_results = pd.read_csv(directory_path + '/%s/%s_%s.csv' % (file, file, method))
                average_values = scenario_results[header[1:]].mean()
                
                new_row = {'Scenario': file, **average_values}
                csv_file_estimation_path_signature = pd.concat([csv_file_estimation_path_signature, pd.DataFrame([new_row])], ignore_index=True)


    # Save the DataFrame to a CSV file
    csv_file_estimation_path_signature.to_csv('%s.csv' % (method), index=False)