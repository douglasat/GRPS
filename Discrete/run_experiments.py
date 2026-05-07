import subprocess
import sys
import os

if __name__ == "__main__":
    directory_path = os.path.dirname(__file__)

    vet = 'optimal'
    # Get all files in the directory
    files = ['blocks-world-%s' %  vet, 'small-sokoban-%s' %  vet, 'satellite-%s' %  vet, 'logistics-%s' %  vet, 'dwr-%s' %  vet, 'depots-%s' %  vet, 'ferry-%s' %  vet, 'driverlog-%s' %  vet, 'miconic-%s' %  vet, 'easy-ipc-grid-%s' %  vet, 'rovers-%s' %  vet, 'zeno-travel-%s' %  vet]
    #files = files[4:]

    for file in files:
        subprocess.run('./test_domain.py "experiments/" %s "vector_inference"' % file, shell=True)