from collections import defaultdict
import os
import numpy as np
import tensorflow as tf
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2, A2C, TRPO

from stable_baselines.common.env_checker import check_env

from environment import AvalonEnv


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def evaluate(model, env, num_episodes):
    """
    Evaluate the model's performance by running `num_episodes` episodes.
    """

    if isinstance(env, VecEnv):
        # Make sure only single environment is passed even if it's a VecEnv
        assert env.num_envs == 1

    reward_so_far = []
    penalties_so_far = []
    agent_game_results = defaultdict(list)

    for _ in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0
        episode_penalties = 0

        while not done:
            # Predict next action using the model
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, _info = env.step(action)

            # Every -ve reward counts as penalty
            if reward < 0:
                episode_penalties += 1

            episode_reward += reward

            if done:  # game over
                if type(_info) is list: # Some models return info in the form of a list
                    info = _info[0]
                else:
                    info = _info
                agent_game_results[info['char_type']].append(info['game_winner'] == info['agent_team'])

        reward_so_far.append(episode_reward)
        penalties_so_far.append(episode_penalties)

    return reward_so_far, penalties_so_far, agent_game_results


class CustomCallback(BaseCallback):
    """
    The callback class. A callback can be configured to execute on different events and
    at certain timestep intervals while training the agent.
    """
    def __init__(self, check_freq, log_dir, num_eval_episodes, target_reward=None, verbose=1):
        """

        :param check_freq: The frequency at which this callback should be triggered.
        :param log_dir: Directory to save the model and other data
        :param num_eval_episodes: Number of episodes for evaluation.
        :param target_reward: If passed, the training stops if the target_reward is reached.
        :param verbose: If verbose is 1, print internal logs (if any).
        """
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.num_eval_episodes = num_eval_episodes
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.target_reward = target_reward

    def _init_callback(self):
        """
        This function is called when the callback class is initialized.
        """
        # Create folders if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def plot_scalar_on_tb(self, tag, value):
        """
        Helper function to create tensorboard graphs.
        :param tag: Name of the graph
        :param value: Value to plot
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

    def _on_step(self):
        """
        This function is called after every env.step during training.
        """
        # Execute the callback logic based on check_freq
        if self.n_calls % self.check_freq == 0:
            # Retrieve historic training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last n episodes
                mean_reward = np.mean(y[-self.num_eval_episodes:])
                print(f"Number of time steps: {self.num_timesteps}")
                print(f"Best mean reward so far: {self.best_mean_reward:.2f} - Running mean reward average: {mean_reward:.2f}")

                # Reward after evaluating on a new environment for num_eval_episodes
                eval_rewards, eval_penalties, eval_agent_results = evaluate(model, model.get_env(), self.num_eval_episodes)
                mean_eval_reward, std_eval_reward = np.mean(eval_rewards), np.std(eval_rewards)

                # Plot these rewards to tensorboard
                self.plot_scalar_on_tb(tag='mean_evaluation_reward', value=mean_eval_reward)
                self.plot_scalar_on_tb(tag='std_evaluation_rewards', value=std_eval_reward)
                self.plot_scalar_on_tb(tag='mean_evaluation_penalties', value=np.mean(eval_penalties))
                self.plot_scalar_on_tb(tag='std_evaluation_rewards', value=np.std(eval_penalties))

                # Calculate win percentage and plot them.
                for char_type, game_results in eval_agent_results.items():
                    total_games = len(game_results)
                    total_wins = np.sum(game_results)
                    win_percent = total_wins / total_games * 100
                    self.plot_scalar_on_tb(tag=f'{char_type.name}_win_percentage', value=win_percent)
                    print(f'Agent won {win_percent}% games out of {total_games} games while taking {char_type.name} role.')

                # New best model, saving it
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Saving best model
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

                # Target achieved, quit training by returning False.
                if self.target_reward and mean_eval_reward > self.target_reward:
                    print(f"Achieved reward of {mean_eval_reward}, halting training")
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                    return False

        return True


if __name__ == "__main__":
    RANDOM_SEED = 90
    MULTIPROCESS = None  # Keep it to None for now. Don't change.
    NUM_EVAL_EPISODES = 100
    LOG_DIR = "./monitor/"
    TENSORBOARD_LOG = "./ppo2_tensorboard/"
    #TENSORBOARD_LOG = None

    # Params for our AvalonEnv
    env_kwargs = dict(num_players=5, enable_logs=False, autoplay=False)
    if MULTIPROCESS:
        # Multiprocessing function, doesn't seem to support monitoring though
        env = make_vec_env(AvalonEnv, n_envs=MULTIPROCESS, seed=RANDOM_SEED, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
    else:
        # Create our AvalonEnv
        env = AvalonEnv(**env_kwargs)
        # check_env confirms compatability of our env with stable-baseline
        check_env(env)
        # Create a vector environment since stable-baseline alogrithms expect it
        env = DummyVecEnv([lambda: Monitor(env, LOG_DIR)])

    # Initializing our callback
    callback = CustomCallback(check_freq=2000, log_dir=LOG_DIR, num_eval_episodes=NUM_EVAL_EPISODES, target_reward=5)


    # Initializing the model
    #model = PPO2(MlpLstmPolicy, env, tensorboard_log=TENSORBOARD_LOG, nminibatches=1) # ineffective
    model = PPO2(MlpPolicy, env, tensorboard_log=TENSORBOARD_LOG) # works nicely, achieved reward of 4+
    #model = A2C(MlpPolicy, env, tensorboard_log=TENSORBOARD_LOG) # works, but not as good as PPO2, reward of 3.6
    #model = TRPO(MlpPolicy, env, tensorboard_log=TENSORBOARD_LOG) # also effective, reward of 5+

    # Training the model, for `total_timesteps` timesteps.
    model.learn(total_timesteps=1000000, callback=callback)
    env.close()
