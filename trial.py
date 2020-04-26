from collections import deque, defaultdict
from flatten_dict import flatten
import numpy as np

from environment import AvalonEnv
from agent import RandomAgent, QTableAgent
from game.enums import CharacterType, PlayerVisibility


def train(num_episodes, env, agent, last_n_plot=50):
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
    reward_so_far = deque(maxlen=last_n_plot)
    penalties_so_far = deque(maxlen=last_n_plot)

    last_n_avg_reward = []
    last_n_avg_penalties = []

    agent_game_results = defaultdict(list)

    for _ in range(num_episodes):
        obs, info = env.reset()
        info['prev_obs'] = None
        reward = 0
        done = False

        episode_reward = 0
        episode_penalties = 0

        while not done:
            action = agent.get_next_action(obs, reward, info)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            info['prev_obs'] = prev_obs

            # Every -ve reward counts as penalty
            if reward < 0:
                episode_penalties += 1

            episode_reward += reward

            if done:  # game over
                agent_game_results[env.agent.char_type].append(env.game.winner == env.agent.team)

        reward_so_far.append(episode_reward)
        penalties_so_far.append(episode_penalties)
        last_n_avg_reward.append(np.mean(reward_so_far))
        last_n_avg_penalties.append(np.mean(penalties_so_far))

        # Uncomment to print running statistics
        # print(f"Results after {num_episodes} episodes:")
        # print(f"Average reward per episode {np.mean(reward_so_far)}")
        # print(f"Average penalties per episode: {np.mean(penalties_so_far)}")
    return last_n_avg_reward, last_n_avg_penalties, agent_game_results


def print_summary(agent_game_results):
    """
    Helper method to print summary of the agent results.
    """
    for char_type, game_results in agent_game_results.items():
        total_games = len(game_results)
        total_wins = np.sum(game_results)
        win_percent = total_wins / total_games * 100
        print(f'Agent won {win_percent}% games out of {total_games} games while taking {char_type.name} role.')


"""
Usage,

- When autoplay is true, all the agent actions are nullified and the game is played automatically as per heuristics
present in the Player class.
- Pass enable_logs as True to debug and see step by step game state changes.
"""
env = AvalonEnv(6, enable_logs=False, autoplay=False)

agent = QTableAgent(env=env)
num_episodes = 500
rewards, penalties, agent_game_results = train(num_episodes, env, agent)

# A nice way to get the q-table and sort it by q-values
# Inspecting the extreme q values (+ve or -ve) help us to see what exactly agent is learning from experiences.
# A nice way to debug and add new features as well.
servant_table = sorted(flatten(agent.q_table[CharacterType.SERVANT]).items(), key=lambda x: x[1])
print_summary(agent_game_results)
