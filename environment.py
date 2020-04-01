import gym

from gym.spaces import Tuple, Discrete

from game_short import AvalonGame

"""
Game state space = Info necessary to make a decision, which includes:

- Personal character type
- Other player character type (in case personal type is evil)
- Current quest number
- Current team proposal number in current quest
- Previous quests status
- Previous proposals


[ctype1 .... ctype5, qno, pno, current_wins, target_wins, action_type_to_take?]

Game config is defined by

Action to take 
[Approve/reject/No-op, Pass/Fail/No-op, Select Team/No-op]

"""

# TODO: Check if we can wrap GoalEnv instead https://github.com/openai/gym/blob/master/gym/core.py#L158

"""
Regarding legal actions: https://github.com/openai/gym/issues/413 
Tuple of discrete action spaces: https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py#L78 and https://github.com/bzier/gym-mupen64plus/blob/3551409db59ccba1912c004c1612b2b1aa2ecc8d/gym_mupen64plus/envs/MarioKart64/discrete_envs.py#L5
Multi discrete action space discussion: https://github.com/openai/universe-starter-agent/issues/75

Ideas: To begin with

- Pass the legal actions info in the info object (needs to be implemented in reset and step method), use that info in the agent
- Make it more strict by removing the legal actions and penalizing the agent for taking non-legal actions


https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
"""


class AvalonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players):
        super(AvalonEnv, self).__init__()

        self.game = AvalonGame(num_players)
        self.action_space = self.action_space = Tuple(
            [Discrete(3), Discrete(3), Discrete(2)]
        )
        self.player_types = [p.char_type.val for p in self.game.players]

        self.observation_space = Tuple((
            Discrete(self.game.num_players),  # Character types
            Discrete(1),  # Quest
            Discrete(1),  # Proposal
            Discrete(1),  # Leader
            #Discrete(self.game.max_quests),  # Current quest
            #Discrete(self.game.max_proposals_allowed),  # Current proposal
        ))

    def step(self, action):
        assert self.action_space.contains(action)
        return self._next_observation()

    def reset(self):
        # Reset all the stuff
        pass
        return self._next_observation()

    def render(self, mode='human'):
        # Bunch of print statements
        pass

    def _next_observation(self):
        obs = [self.player_types, self.game.current_quest, self.game.current_proposal_number, self.game.current_leader]
        return obs

    def _take_action(self, action):
        # Do different stuff based on the action type
        pass


def sample_run():
    env = AvalonEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        print("Observation", obs)
        print("Reward", rewards)
        print("Done status", done)
        print("Info", info)
        env.render()
