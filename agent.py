from collections import defaultdict
import random


class Agent:
    def __init__(self, env):
        self.env = env

    def get_next_action(self, obs, reward, info):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__(**kwargs)

    def get_next_action(self, obs, reward, info):
        return self.env.action_space.sample()


class QTableAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.6, epsilon=0.1, **kwargs):
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

    def get_next_action(self, new_obs, reward, info):
        q_table = self.get_q_table(info['char_type'])
        old_obs = info['prev_obs']
        action_taken = info['prev_action']
        if action_taken:
            old_value = q_table[old_obs][action_taken]
            #TODO: use numpy for speedup
            next_max = max(q_table[new_obs].values()) if len(q_table[new_obs].values()) > 0 else 0.0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            q_table[old_obs][action_taken] = new_value

        return self._select_next_action(q_table, new_obs)

    def _select_next_action(self, q_table, obs):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()  # Explore action space
        else:
            #TODO: use numpy for speedup
            q_vals = q_table[obs]
            action = max(q_vals, key=q_vals.get, default=self.env.action_space.sample())  # Exploit learned values
        return action
