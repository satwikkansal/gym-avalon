from collections import deque, defaultdict
from flatten_dict import flatten
import pickle

import numpy as np

from environment import AvalonEnv
from agent import RandomAgent, QTableAgent
from game.enums_and_config import CharacterType

BEST_MEAN_REWARD = -float('inf')


def evaluate(agent, env, num_episodes):
    """
    Evaluate the agent, on a new envrionment instance.
    """
    reward_so_far = []
    penalties_so_far = []
    agent_game_results = defaultdict(list)

    for _ in range(num_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0
        episode_penalties = 0
        _info = None

        while not done:
            action = agent.predict(obs, _info)
            obs, reward, done, _info = env.step(action)

            if type(_info) is list:
                info = _info[0]
            else:
                info = _info

            episode_reward += reward
            episode_penalties += info['num_penalties']

            if done:  # game over
                agent_game_results[info['char_type']].append(info['game_winner'] == info['agent_team'])

        reward_so_far.append(episode_reward)
        penalties_so_far.append(episode_penalties)

    return reward_so_far, penalties_so_far, agent_game_results


def callback(env, model, save_path, num_eval_episodes=100, target_reward=None):
    """
    Callback method that, evaluates performance, prints out metrics, saves the best models, and
    terminates training if target reward is achieved.
    """
    eval_rewards, eval_penalties, eval_agent_results = evaluate(model, env, num_eval_episodes)
    mean_eval_reward, std_eval_reward, mean_eval_penalties = np.mean(eval_rewards), np.std(eval_rewards), np.mean(eval_penalties)
    print()
    print(f"Average reward per episode {np.mean(eval_rewards)}")
    print(f"Average penalties per episode: {np.mean(eval_penalties)}")

    win_percent_by_ctype = {}
    for char_type, game_results in eval_agent_results.items():
        total_games = len(game_results)
        total_wins = np.sum(game_results)
        win_percent = total_wins / total_games * 100
        win_percent_by_ctype[char_type] = win_percent
        print(f'Agent won {win_percent}% games out of {total_games} games while taking {char_type.name} role.')

    global BEST_MEAN_REWARD
    # New best model, you could save the agent here
    if mean_eval_reward > BEST_MEAN_REWARD:
        BEST_MEAN_REWARD = mean_eval_reward
        # Saving best model
        print(f"Saving new best model to {save_path}")
        #TODO: Enable save later on
        #save(model.q_table, save_path)

    if target_reward and mean_eval_reward > target_reward:
        print(f"Achieved reward of {mean_eval_reward}, halting training")
        print(f"Saving new best model to {save_path}")
        save(model.q_table, save_path)
        return False  # Signal to terminate training

    return win_percent_by_ctype, mean_eval_reward, std_eval_reward, mean_eval_penalties


def train(num_episodes, env, agent, target_reward=4, last_n_plot=100, callback_every=5000):
    """
    The method to train the agent.
    :param num_episodes: number of episodes (game instances) to train the agent for
    :param env: The environment in which to train the agent.
    :param agent: The agent instance.
    :param last_n_plot: Number of latest metrics to track as history.
    :return:
    """
    # Store last few metrics here
    # These will be useful for plotting and visualizing results later on.
    episode_rewards, mean_eval_rewards, std_eval_rewards, mean_eval_penalties = [], [], [], []
    agent_game_results = defaultdict(list)

    for curr_episode in range(num_episodes):
        if curr_episode % callback_every == 0:
            win_percent_map, mean_eval_reward, std_eval_reward, mean_eval_penalty = callback(env, agent, "q_table.pickle", last_n_plot, target_reward)
            mean_eval_rewards.append(mean_eval_reward)
            std_eval_rewards.append(std_eval_reward)
            mean_eval_penalties.append(mean_eval_penalty)

            if win_percent_map is False:
                break
            else:
                for ctype, percent in win_percent_map.items():
                    agent_game_results[ctype].append(percent)

        obs, info = env.reset(external=False)
        info['prev_obs'] = None
        reward = 0
        done = False

        episode_reward = 0

        while not done:
            action = agent.get_next_action(obs, reward, info)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            info['prev_obs'] = prev_obs
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return episode_rewards, mean_eval_rewards, std_eval_rewards, mean_eval_penalties, agent_game_results


def save(model, save_path):
    """
    Save the q-table as pickle
    """
    with open(save_path, "wb") as f:
        d = dict(model)
        pickle.dump(d, f)


if __name__ == "__main__":
    """
    Usage,

    - When autoplay is true, all the agent actions are nullified and the game is played automatically as per heuristics
    present in the Player class.
    - Pass enable_logs as True to debug and see step by step game state changes.
    """
    env = AvalonEnv(5, enable_logs=False, autoplay=True, enable_penalties=False)

    agent = QTableAgent(env=env)
    num_episodes = 150000
    rewards, mean_eval_rewards, std_eval_rewards, mean_eval_penalties,  agent_game_results = train(num_episodes, env, agent)

    # A nice way to get the q-table and sort it by q-values
    # Inspecting the extreme q values (+ve or -ve) help us to see what exactly agent is learning from experiences.
    # A nice way to debug and add new features as well.
    servant_table = sorted(flatten(agent.q_table[CharacterType.SERVANT]).items(), key=lambda x: x[1])
    servant_table_reversed = list(reversed(servant_table))


