class Agent:
    def get_next_action(self, obs, reward, info, action_space):
        raise NotImplementedError


class RandomAgent(Agent):
    def get_next_action(self, obs, reward, info, action_space):
        return action_space.sample()
