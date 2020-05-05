from collections import defaultdict
import numpy as np
import random


def float_dd():
    return defaultdict(float)

class Agent:
    def __init__(self, env):
        self.env = env

    def get_next_action(self, obs, reward, info):
        raise NotImplementedError

    def _sample(self, total_players, players_to_choose):
        """
        Custom sample function. The action space for team
        selection is MultiBinary(5) but the validity of actions
        depend on the players_to_choose (Ex [1 1 1 1 1] i.e. all 5 players in team
        is an invalid action if we need only 3 players), that's why a custom function
        that can give sample actions that are also valid.
        """
        action_space = self.env.action_space
        return tuple(
            [
                self._get_sample_team_selection_action(total_players, players_to_choose),
                action_space.spaces[1].sample(),
                action_space.spaces[2].sample()
            ]
        )

    def _get_sample_team_selection_action(self, total_players, players_to_choose):
        # Create an array of zeroes and ones indicating selection
        arr = np.array([0] * (total_players - players_to_choose) + [1] * players_to_choose)
        # Shuffle inplace
        np.random.shuffle(arr)
        return arr


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__(**kwargs)

    def get_next_action(self, obs, reward, info):
        return self._sample(info['num_players'], info['quest_team_size'])


class QTableAgent(Agent):
    """
    Every Character type has it's own set of q-values since the dynamics of games
    are different for different character types.

    The structure of the q_table in this class is as follows

    {
        char_type_1: {
            state_1: {
                action_1: q_value,
                action_2: q_value
            },
            state_2: {
                action_1: q_value,
                action_2: q_value
            },
        },
        char_type_2: {
            state_1: {
                action_1: q_value,
                action_2: q_value
            },
            state_2: {
                action_1: q_value,
                action_2: q_value
            },
        }
    }
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, **kwargs):
        super(QTableAgent, self).__init__(**kwargs)
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _init_q_table_for_char_type():
        # Default value of 0 for every state-action pair
        return defaultdict(float_dd)

    def get_q_table(self, char_type):
        """
        Helper method to get Q-table for a character type. Creates one if doesn't exists
        already.
        """
        if char_type not in self.q_table:
            self.q_table[char_type] = self._init_q_table_for_char_type()
        return self.q_table[char_type]

    def get_next_action(self, new_obs, reward, info):
        q_table = self.get_q_table(info['char_type'])

        new_obs = tuple(new_obs)
        action_taken = info['prev_action']

        if action_taken is not None:
            # Team selection Action needs to be casted to a tuple to store in a dictionary
            # action_taken = (tuple(action_taken[0]), action_taken[1], action_taken[2])
            # Updating the q_values as per bellman equation
            action_taken = tuple(action_taken)
            old_obs = tuple(info['prev_obs'])
            old_value = q_table[old_obs][action_taken]
            next_max = max(q_table[new_obs].values()) if len(q_table[new_obs].values()) > 0 else 0.0
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            q_table[old_obs][action_taken] = new_value

        return self.predict(new_obs, info, False)

    def predict(self, obs, info, direct=True):
        """
        Select next action based on epsilon-greedy method.
        TODO: Use better strategy for exploration than the epsilon-greedy method.
        """
        random_action = self.env.action_space.sample()
        if info is None:
            return random_action

        q_table = self.get_q_table(info['char_type'])
        random_val = random.uniform(0, 1)
        #random_action = self._sample(info['num_players'], info['quest_team_size'])
        if not direct and random_val < self.epsilon:
            action = random_action  # Explore action space
        else:
            q_vals = q_table[tuple(obs)]
            action = max(q_vals, key=q_vals.get, default=random_action)  # Exploit learned values
            #action = (np.array(action[0]), action[1], action[2])
            action = list(action)
        return action
