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
 
### Tensorboard

The directory is specified in `TENSORBOARD_LOG` variable in `baseline_trial.py` file. The default folder is `ppo2_tensorboard`.

To lauch tensorboard you need to provide path to this directory.

```shell script
$ tensorboard --logdir ./ppo2_tensorboard/
```

This should give you a link that you can browse in your browser. 