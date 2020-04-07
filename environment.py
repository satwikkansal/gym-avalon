import gym

from gym.spaces import Tuple, Discrete

from game.enums import ActionType
from game.avalon import AvalonGame

"""
Game state space
----------------
It is the Info necessary to make a decision, which includes:

- Personal character type
- Other player character type (in case personal type is evil)
- Current quest number
- Current team proposal number in current quest
- Previous quests status and previous proposals (most probably at the player level)


Actions space
-------------
Three kinds of actions: Team Selection, Team Approval and Quest voting
[Select Team/No-op, Approve/reject/No-op, Pass/Fail/No-op]


References
----------
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
            Discrete(self.num_players),  # Current team
        ))

    def _initialize_game(self, num_players):
        self.game = AvalonGame(num_players)
        self.player_types = [p.char_type.value for p in self.game.players]
        self.agent = list(filter(lambda x: x.player_id == 0, self.game.players))[0] # Agent is the player with ID 0
        self.quests_history = [] # Useful for computing the reward
        self.quest_reward_count = 0

    def _convert_game_feedback_to_observation(self, feedback):
        return [
            self.player_types,
            feedback.quest_number,
            feedback.proposal_number,
            feedback.leader,
            feedback.current_team
        ]

    def step(self, action):
        """
        Executes an action in the game, and returns the reward and next observation.
        Note that after execution agent's actions, all other auto-actions (by non-agent players)
        are executed before computing the next observation and reward for the agent.
        """
        assert self.action_space.contains(action)
        action_type = self.game.current_quest.current_action_type
        agent_action_feedback = self._take_action(action, action_type)
        feedback = self._get_agent_feedback(agent_action_feedback)
        obs = self._convert_game_feedback_to_observation(feedback)
        reward = self.compute_reward(feedback, action, action_type)
        done = feedback.game_winner is not None
        info = {'next_action': feedback.action_type}
        return obs, reward, done, info

    def compute_reward(self, feedback, action, action_type):
        """
        Compute reward for the action taken.

        Ideas:
        - Penalize for illegal actions?
        - Mission loose results in -ve reward
        - Mission win results in +ve reward
        - Game win / loose rewards
        """
        reward = 0

        # Reward for winning / losing the quests
        if self.quest_reward_count < len(self.quests_history):
            last_quest = self.quests_history[-1]
            if last_quest.quest_winner == self.agent.team:
                reward += 1
            else:
                reward -= 1
            self.quest_reward_count += 1

        # Reward for winning / losing the game
        if feedback.game_winner is not None:
            if feedback.game_winner == self.agent.team:
                reward += 1
            else:
                reward -= 1

        # TODO: Add logic for action penalizing here.

        return reward

    def reset(self):
        # Reset all the stuff
        self._initialize_game(self.num_players)
        feedback = self._get_agent_feedback()
        return self._convert_game_feedback_to_observation(feedback)

    def render(self, mode='human'):
        print(self.game)

    def _get_agent_feedback(self, prev_feedback=None):
        """
        Runs the game until there's an event relevant for the agent, and then
        returns the feedback.
        """
        feedback = prev_feedback if prev_feedback is not None else self.game.run()
        while not(feedback.action_required or feedback.game_winner):
            self.render()
            if feedback.initiate_new_quest:
                assert feedback.quest_winner
                self.quests_history.append(self.game.current_quest)
                self.game.initialize_new_quest()
            feedback = self.game.run()
        return feedback

    def _take_action(self, action, action_type):
        """
        Take an action based on the action type and action
        vector provided by the agent. Return the feedback.
        """
        action_to_value = self.action_value_map[action_type]
        if action_type == ActionType.TEAM_SELECTION:
            # Choose team randomly for now
            feedback = self.game.run(self.agent, override_choice=None)
        elif action_type == ActionType.TEAM_APPROVAL:
            relevant_action = action[1]
            action_to_take = action_to_value[relevant_action]
            feedback = self.game.run(self.agent, override_choice=action_to_take)
        elif action_type == ActionType.QUEST_VOTE:
            relevant_action = action[2]
            action_to_take = action_to_value[relevant_action]
            feedback = self.game.run(self.agent, override_choice=action_to_take)
        return feedback
