import gym
from gym.spaces import Tuple, Discrete, MultiBinary, MultiDiscrete
import numpy as np

from game.enums_and_config import ActionType, PlayerVisibility, Team, team_selection_move_map
from game.avalon import AvalonGame

"""
Game state space
----------------
It is the Info necessary to make a decision, which includes:

- Personal character type
- Other player character type (in case personal type is evil)
- Previous quests status and previous proposals (most probably at the player level)

These three factors are combined into PlayerVisibility. Other factors include,
- Current quest number
- Current team proposal number in current quest


Actions space
-------------
Three kinds of actions: Team Selection, Team Approval and Quest voting
[Select Team, No-Op/Approve/reject, No-Op/Pass/Fail]


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

    def __init__(self, num_players, enable_logs, autoplay=False, enable_penalties=True):
        """
        :param num_players: Number of players to set up the game for. Passed as an arg to game initialization.
        :param enable_logs: Pass enable_logs as True to debug and see step by step game state changes.
        :param autoplay: When autoplay is true, all the agent actions are nullified and the game is played automatically as per heuristics
                         present in the Player class.
        """
        super(AvalonEnv, self).__init__()
        self.enable_logs = enable_logs
        self.autoplay = autoplay
        self.enable_penalties = enable_penalties

        self.num_players = num_players
        self._initialize_game(num_players, enable_logs)


        # Commenting because Tuple space isn't supported by many models.
        """
        # See action_value_map for the behavior of various action values
        self.action_space = Tuple(
            [
                MultiBinary(self.num_players),  # Select players
                Discrete(3),  # No op/ Approve / Reject
                Discrete(3),  # No op / Pass / Fail
            ]
        )

        self.observation_space = Tuple((
            Discrete(len(PlayerVisibility)),  # Explained in the top comments
            Discrete(len(ActionType)),  # Action type
            Discrete(5),  # Quest
            Discrete(5),  # Proposal #TODO: Fixen(team_selection_move_map[self.num_players])
            Discrete(5),  # Leader
        ))
        """

        # Use MultiDiscrete action and observation spaces instead
        self.action_space = MultiDiscrete([len(team_selection_move_map[self.num_players]), 3, 3])
        
        # Player visibilites, action type, current quest, proposal number, current leader
        obs_space = [len(PlayerVisibility)] * self.num_players
        obs_space += [len(ActionType), self.game.max_quests, self.num_players, self.num_players]

        self.observation_space = MultiDiscrete(
            obs_space
        )

    def _initialize_game(self, num_players, enable_logs):
        self.game = AvalonGame(num_players, enable_logs=enable_logs)
        self.player_types = tuple(p.char_type.value for p in self.game.players)
        self.agent = list(filter(lambda x: x.player_id == 0, self.game.players))[0]  # Agent is the player with ID 0
        # Useful for computing the reward for winning / losing a quest
        self.quests_history = []
        self.quest_reward_count = 0

    def _convert_game_feedback_to_observation(self, feedback, prev_action, penalties):
        """
        The method that creates the observation and info object for the environment.
        It uses feedback received from the game to do so.
        """
        visibilities = []

        for player in self.game.players:
            # Is the player in current team?
            in_team = player in feedback.current_team
            total_missions = len(player.mission_history)
            # How many passed and failed missions is the player part of?
            passed_missions = sum(player.mission_history)
            failed_missions = total_missions - passed_missions
            if self.agent.team is Team.EVIL:
                # Evil agent has full visibility about the player's team
                visibilities.append(PlayerVisibility[(player.team, in_team, passed_missions, failed_missions)])
            else:
                if player is self.agent:
                    visibilities.append(PlayerVisibility[(Team.GOOD, in_team, passed_missions, failed_missions)])
                else:
                    # Good agent has no visibility about the player's team
                    visibilities.append(PlayerVisibility[(Team.UNKNOWN, in_team, passed_missions, failed_missions)])

        # Pack into an observation
        # Note this structure should conform to the one defined in self.observation_space
        # Commenting this because no longer using Tuple spaces
        """
        obs = (
            tuple(visibilities),
            feedback.action_type.value,
            feedback.quest_number,
            feedback.proposal_number,
            feedback.leader,
        )
        """
        # Adjusting as per MultiDiscrete space
        obs = np.concatenate([
             visibilities,
            [feedback.action_type.value, feedback.quest_number, feedback.proposal_number, feedback.leader]
            ])

        # Extra Info that needs to be passed to agent to decide stuff.
        # This info is however not a part of state space, so will only be helpful
        # in making immediate decisions.
        info = {
            'next_action': feedback.action_type,
            'char_type': self.agent.char_type,
            'prev_action': prev_action,
            'num_players': self.game.num_players,
            'quest_team_size': feedback.quest_team_size,
            'game_winner': feedback.game_winner,
            'agent_team': self.agent.team,
            'num_penalties': penalties
        }

        return obs, info

    def step(self, action):
        """
        Executes an action in the game, and returns the reward and next observation.
        Note that after execution agent's actions, all other auto-actions (by non-agent players)
        are executed before computing the next observation and reward for the agent.
        """
        # Disabling because tuple values aren't supported by contains in MultiBinary spcaes.
        # assert self.action_space.contains(action)
        action_type = self.game.current_quest.current_action_type
        agent_action_feedback = self._take_action(action, action_type)
        feedback = self._get_agent_feedback(agent_action_feedback)
        reward, penalties = self.compute_reward(feedback, action, action_type)
        obs, info = self._convert_game_feedback_to_observation(feedback, action, penalties)
        done = feedback.game_winner is not None

        return obs, reward, done, info

    def compute_reward(self, feedback, action, action_type):
        """
        Compute reward for the action taken.

        Ideas:
        - Penalize for illegal actions
        - Reward for legal actions
        - Mission loose results in -ve reward
        - Mission win results in +ve reward
        - Game win results in +ve reward
        - Game loss results in -ve reward
        """
        reward = 0
        num_penalties = 0

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
                reward += 2
            else:
                reward -= 2

        relevant_action = self._get_relevant_value_for_action(action, action_type)

        if relevant_action is None and self.enable_penalties:
            # Penalize for not following the relevant action
            reward -= 0.5
            num_penalties += 1
        else:
            reward += 0.1

        for at in ActionType:
            if at == ActionType.TEAM_SELECTION:
                continue
            if at != action_type and action[at.value] and self.enable_penalties:
                reward -= 0.5
                num_penalties += 1
            else:
                reward += 0.1

        return reward, num_penalties

    def reset(self, external=True):
        # Reset all the stuff
        self._initialize_game(self.num_players, self.enable_logs)
        feedback = self._get_agent_feedback()
        obs, info = self._convert_game_feedback_to_observation(feedback, None, 0)
        if external:
            return obs
        return obs, info

    def render(self, mode='human'):
        if self.enable_logs:
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

    def _get_relevant_value_for_action(self, action, action_type):
        """
        A method that translates the action conforming to self.action_space
        to a corresponding value that can be passed to the game.
        """
        action_value_map = {
            ActionType.TEAM_SELECTION: team_selection_move_map[self.num_players],  # The vector value should
            ActionType.TEAM_APPROVAL: {
                0: None,
                1: True,
                2: False
            },
            ActionType.QUEST_VOTE: {
                0: None,
                1: True,
                2: False
            }
        }

        relevant_action = action_value_map[action_type][action[action_type.value]]

        if action_type is ActionType.TEAM_SELECTION:
            # relevant_action = (action[action_type.value].nonzero())[0]
            if len(relevant_action) != self.game.current_quest.team_size:
                relevant_action = None

        return relevant_action

    def _take_action(self, action, action_type):
        """
        Take an action based on the action type and action
        vector provided by the agent. Return the feedback.
        """
        relevant_action = None
        if not self.autoplay:
            relevant_action = self._get_relevant_value_for_action(action, action_type)
        return self.game.run(self.agent, override_choice=relevant_action)

