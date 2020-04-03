import gym

from gym.spaces import Tuple, Discrete

from game_short import AvalonGame, ActionType

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

Actions to take 
[Approve/reject/No-op, Pass/Fail/No-op, Select Team/No-op]

References:
Regarding legal actions: https://github.com/openai/gym/issues/413 
Tuple of discrete action spaces: https://github.com/openai/gym/blob/master/gym/envs/algorithmic/algorithmic_env.py#L78 and https://github.com/bzier/gym-mupen64plus/blob/3551409db59ccba1912c004c1612b2b1aa2ecc8d/gym_mupen64plus/envs/MarioKart64/discrete_envs.py#L5
Multi discrete action space discussion: https://github.com/openai/universe-starter-agent/issues/75
Check if we can wrap GoalEnv instead https://github.com/openai/gym/blob/master/gym/core.py#L158
Blackjack https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
"""


class AvalonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    action_value_map = {
        ActionType.TEAM_SELECTION: {},
        ActionType.TEAM_APPROVAL: {
            0: True,
            1: False,
            2: None
        },
        ActionType.QUEST_VOTE: {
            0: True,
            1: False,
            2: None
        }
    }

    def __init__(self, num_players):
        super(AvalonEnv, self).__init__()

        self.num_players = num_players
        self._initialize_game(num_players)

        self.action_space = self.action_space = Tuple(
            [
                Discrete(2),  # Randomly select team / No op
                Discrete(3),  # Pass / Fail / No op
                Discrete(3),  # Approve / Reject / No op
            ]
        )
        self.observation_space = Tuple((
            Discrete(self.num_players),  # Character types
            Discrete(1),  # Quest
            Discrete(1),  # Proposal
            Discrete(1),  # Leader
        ))

    def _initialize_game(self, num_players):
        self.game = AvalonGame(num_players)
        self.player_types = [p.char_type.value for p in self.game.players]
        self.agent = list(filter(lambda x: x.player_id == 0, self.game.players))[0]

    def step(self, action):
        assert self.action_space.contains(action)
        self._take_action(action)
        obs, feedback = self._next_observation()
        reward = self.compute_reward(obs, action)
        done = feedback is True
        info = {'next_action': feedback}
        return obs, reward, done, info

    def compute_reward(self, obs, action):
        """
        Compute reward for the action taken.
        """
        return 0

    def reset(self):
        # Reset all the stuff
        self._initialize_game(self.num_players)
        return self._next_observation()

    def render(self, mode='human'):
        print(self.game)

    def _next_observation(self):
        feedback = None
        while feedback is None:
            feedback = self.game.run()
            self.render()
        obs = [self.player_types, self.game.current_quest, self.game.current_proposal_number, self.game.current_leader]
        return obs, feedback

    def _take_action(self, action):
        action_to_value = self.action_value_map[self.game.current_action_type]
        if self.game.current_action_type == ActionType.TEAM_SELECTION:
            # Choose team randomly for now
            self.game.make_team_selection_move(self.agent)
        elif self.game.current_action_type == ActionType.TEAM_APPROVAL:
            relevant_action = action[1]
            action_to_take = action_to_value[relevant_action]
            self.game.make_team_approval_move(self.agent, override_choice=action_to_take)
        elif self.game.current_action_type == ActionType.QUEST_VOTE:
            relevant_action = action[2]
            action_to_take = action_to_value[relevant_action]
            self.game.make_quest_vote_move(self.agent, override_choice=action_to_take)
