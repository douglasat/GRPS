
# Code Instructions for the Experiments of the Paper:  
## *Online Goal Recognition using Path Signature and Dynamic Time Warping*
To run the experiments, follow these steps:

# Install requirements

Symk Top-K planner: https://github.com/speckdavid/symk
PPDL parser: https://github.com/pucrs-automated-planning/pddl-parser

Navigate to folder and install requirements:
```bash
cd /code
pip install -r requirements.txt
```
# Running the Experiments

All experiments are run using the script `compute_experiments.py`, located in the `/code/Continuous` directory.
## Example (Continuous Domain):
```bash
python3 compute_experiments.py -method gprs -par 10 -m 0.0 -p 0.0 -k 5
```

All experiments are run using the script `compute_experiments.py`, located in the `/code/Discrete` directory.

## Example (Discrete Domain):
```bash
python3 compute_experiments.py -method gprs -m 0.0 -p 0.0 -k 5
```

# Script Arguments

| Argument     | Description |
|--------------|-------------|
| `-method`    | **(str)** Method name to use:<br>• `gprs`: Goal Recognition with Path Signature<br>• `gprs_dtw`: GRPS with Dynamic Time Warping |
| `-par`       | **(int)** Number of parallel processes to use (e.g., `10`) |
| `-m`         | **(float)** Merge threshold. Use `0` to disable merging |
| `-p`         | **(float)** Prune threshold. Use `0` to disable pruning |
| `-k`         | **(int)** Top-K parameter: how many top-ranked trajectories to consider (e.g., `1`, `15`) |

---

# Output Format

After running the script, a `.csv` file containing the experiment results will be generated and saved in the `results/` directory.

- The CSV file includes performance metrics (e.g., PPV) for each problem and parameter combination used in the experiment.

---