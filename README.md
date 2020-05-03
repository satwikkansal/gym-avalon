## Instructions to run

```shell script
$ pip install -r requirements.txt
```

The q-learning training logic is present `q_learning_trial.py` file.
The PPO, TRPO, and A2C training logic is present in `baseline_trial.py` file.
 Both the files can be run from the command line.

Separate because, tabular q-learning agent doesn't use tensorflow and trains slightly different
 (has a different model for every character type) from other algorithms in baseline.
 
The `training.ipynb` file is jupyter notebook plotting graphs for q-learning agent.
 
