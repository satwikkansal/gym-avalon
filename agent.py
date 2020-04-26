from collections import defaultdict
import numpy as np
import random

from game.enums import ActionType


class Agent:
    def __init__(self, env):
        self.env = env

    def get_next_action(self, obs, reward, info):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__(**kwargs)

    def get_next_action(self, obs, reward, info):
        return self.env.action_space.sample() # TODO: Use custom sample function


class QTableAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.2, **kwargs):
        super(QTableAgent, self).__init__(**kwargs)
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _init_q_table_for_char_type():
        return defaultdict(lambda: defaultdict(float))

    def get_q_table(self, char_type):
        if char_type not in self.q_table:
            self.q_table[char_type] = self._init_q_table_for_char_type()
        return self.q_table[char_type]

    def _sample(self, total_players, players_to_choose):
        """
        #TODO: Move to super class
        :param total_players:
        :param players_to_choose:
        :return:
        """
        action_space = self.env.action_space
        return tuple(
            [
                self._get_sample_team_selection_action(total_players, players_to_choose),
                #action_space.spaces[0].sample(),
                action_space.spaces[1].sample(),
                action_space.spaces[2].sample()
            ]
        )

    def _get_sample_team_selection_action(self, total_players, players_to_choose):
        arr = np.array([0] * (total_players - players_to_choose) + [1] * players_to_choose)
        np.random.shuffle(arr)
        return arr

    def get_next_action(self, new_obs, reward, info):
        q_table = self.get_q_table(info['char_type'])
        old_obs = info['prev_obs']
        action_taken = info['prev_action']

        if action_taken:
            action_taken = (tuple(action_taken[0]), action_taken[1], action_taken[2])
            old_value = q_table[old_obs][action_taken]
            next_max = max(q_table[new_obs].values()) if len(q_table[new_obs].values()) > 0 else 0.0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            q_table[old_obs][action_taken] = new_value

        return self._select_next_action(q_table, new_obs, info)

    def _select_next_action(self, q_table, obs, info):
        random_val = random.uniform(0, 1)
        random_action = self._sample(info['num_players'], info['quest_team_size'])
        if random_val < self.epsilon:
            action =  random_action # Explore action space
        else:
            q_vals = q_table[obs]
            action = max(q_vals, key=q_vals.get, default=random_action)  # Exploit learned values
            action = (np.array(action[0]), action[1], action[2])
        return action
