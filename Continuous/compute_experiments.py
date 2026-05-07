import subprocess
import sys
import os
import argparse

def process_input():
    """Main function to handle inputs and call processing logic."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inputs")
    parser.add_argument(
        "-method", "--method", type=str, required=True, help="The first parameter (str)."
    )
    parser.add_argument(
        "-par", "--parallel", type=int, required=True, help="The first parameter (int)."
    )
    parser.add_argument(
        "-m", "--merge", type=float, required=True, help="The second parameter (float)."
    )
    parser.add_argument(
        "-p", "--prune", type=float, required=True, help="The third parameter (float)."
    )
    parser.add_argument(
        "-k", "--topk", type=int, required=True, help="The second parameter (int)."
    )
    
    args = parser.parse_args()
    return args.method, args.parallel, args.merge, args.prune, args.topk


if __name__ == "__main__":
    method, par, merge, prune, topk = process_input()
    directory_path = os.path.dirname(__file__)

     # Get all files in the directory
    scenarios = [file[:-4] for file in os.listdir(directory_path + '/scenarios') if os.path.isfile(os.path.join(directory_path + '/scenarios', file))]
    scenarios = sorted(scenarios)
   
    for scenario in scenarios:
        if 'grps' == method:
            print('Computing GRPS')
            subprocess.run('python3 grps.py -map %s -par %d -m %f -p %f -k %d' % (scenario, par, merge, prune, topk), shell=True)
   
        if 'grps_dtw' == method:
            print('Computing GRPS+DTW')
            subprocess.run('python3 grps_dtw.py -map %s -par %d -m %f -p %f -k %d' % (scenario, par, merge, prune, topk), shell=True)
   
    subprocess.run('python3 data_aggregation.py -method %s' % (method), shell=True)
