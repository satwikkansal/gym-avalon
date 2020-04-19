from collections import deque, defaultdict
import numpy as np

from environment import AvalonEnv
from agent import RandomAgent, QTableAgent
from game.enums import CharacterType


def train(num_episodes, env, agent, last_n_plot=50):
    reward_so_far = deque(maxlen=last_n_plot)
    penalties_so_far = deque(maxlen=last_n_plot)

    last_n_avg_reward = []
    last_n_avg_penalties = []

    agent_game_results = defaultdict(list)

    for _ in range(num_episodes):
        obs = env.reset()
        reward = 0
        info = {
            'char_type': env.agent.char_type,
            'prev_obs': obs,
            'prev_action': None
        }
        done = False

        episode_reward = 0
        episode_penalties = 0

        while not done:
            action = agent.get_next_action(obs, reward, info)
            prev_obs = obs
            obs, reward, done, info = env.step(action)
            info['prev_obs'] = prev_obs

            if reward <= -1:
                episode_penalties += 1

            episode_reward += reward

            if done:
                agent_game_results[env.agent.char_type].append(env.game.winner == env.agent.team)

        reward_so_far.append(episode_reward)
        penalties_so_far.append(episode_penalties)
        last_n_avg_reward.append(np.mean(reward_so_far))
        last_n_avg_penalties.append(np.mean(penalties_so_far))

    #     print(f"Results after {num_episodes} episodes:")
    #     print(f"Average reward per episode {np.mean(reward_so_far)}")
    #     print(f"Average penalties per episode: {np.mean(penalties_so_far)}")
    return last_n_avg_reward, last_n_avg_penalties, agent_game_results


def print_summary(agent_game_results):
    for char_type, game_results in agent_game_results.items():
        total_games = len(game_results)
        total_wins = np.sum(game_results)
        win_percent = total_wins / total_games * 100
        print(f'Agent won {win_percent}% games out of {total_games} games while taking {char_type.name} role.')


env = AvalonEnv(5, enable_logs=False, autoplay=False)
agent = QTableAgent(env=env)
num_episodes = 20000
rewards, penalties, agent_game_results = train(num_episodes, env, agent)
servant_table = agent.q_table[CharacterType.SERVANT]
print_summary(agent_game_results)

