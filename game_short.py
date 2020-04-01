from collections import namedtuple
from enum import Enum

import random


"""
Evil aka Minions of mordred
Good aka Servants of mordred

Merlin: The person who knows who evils are


If evils guess at the end of the game, who the merlin is, the evils win.
"""


# Number of players: [[good_size, evil_size], [mission team size for each round]]
player_config = namedtuple('player_config', ['team_split', 'quest_size'])

game_player_configs = {
    5: player_config([3, 2], [2, 3, 2, 3, 3]),
    6: player_config([4, 2], [2, 3, 4, 3, 4]),
    7: player_config([4, 3], [2, 3, 3, 4, 4]),
    8: player_config([5, 3], [3, 4, 4, 5, 5]),
    9: player_config([6, 3], [3, 4, 4, 5, 5]),
    10: player_config([6, 4], [3, 4, 4, 5, 5]),
    11: player_config([7, 4], [4, 5, 4, 5, 5])
}


class CharacterType(Enum):
    SERVANT = 1
    MERLIN = 2
    MINION = 3


class ActionType(Enum):
    TEAM_SELECTION = 1
    TEAM_APPROVAL = 2
    QUEST_VOTE = 3


class Team(Enum):
    GOOD = 1
    EVIL = 2


class Player:
    def __init__(self, player_id, char_type):
        self.player_id = player_id
        self.char_type = char_type
        if char_type == CharacterType.MINION:
            self.team = Team.EVIL
        else:
            self.team = Team.GOOD

    # Action
    def approve_or_reject_quest(self):
        # TODO: Add heuristics here
        return True

    def is_evil(self):
        """
        :return: True or False depending upon the character type
        """
        return self.team is Team.EVIL

    def success_or_fail(self):
        if self.is_evil():
            # TODO: Add heuristics here
            return False
        else:
            return True

    def is_agent(self):
        return self.player_id == 0


class AvalonGame:
    def __init__(self, num_players, max_quests=5, max_proposals_allowed=5, enable_agent=False):
        self.num_players = num_players
        self.players = self.initialize_players()
        self.enable_agent = enable_agent  # for now, agent is always player 0

        self.max_quests = max_quests
        self.max_proposals_allowed = max_proposals_allowed
        # Number of quests needed to be won by a team to win the game
        self.win_target = max_quests // 2 + 1
        self.quest_sizes = game_player_configs[num_players].quest_size

        self.current_leader = 0
        self.current_quest = 0
        self.current_proposal_number = 0
        self.current_team = []
        self.approvals = 0

        self.good_team_wins = 0
        self.evil_team_wins = 0

        self.current_action_type = ActionType.TEAM_SELECTION
        self.pending_turns = {
            ActionType.TEAM_SELECTION: self.generate_team_selection_turns()
        }

        self.winner = None

    def initialize_players(self):
        """
        Assigns player types based on the number of players and game_player_config.s
        :return:
        """
        # Random sampling for random assignment of character types
        player_order = random.sample(range(self.num_players), self.num_players)
        good_count, evil_count = game_player_configs[self.num_players].team_split

        assert good_count + evil_count == self.num_players

        players = []
        for player_id in player_order:
            if player_id + 1 < good_count:
                player = Player(player_id, CharacterType.SERVANT)
            elif player_id + 1 == good_count:
                player = Player(player_id, CharacterType.MERLIN)
            else:
                player = Player(player_id, CharacterType.MINION)
            players.append(player)

        return players

    def generate_team_selection_turns(self):
        return [self.players[(self.current_leader + i) % self.num_players] for i in range(self.num_players)]

    def run(self):
        if self.good_team_wins == self.win_target or self.evil_team_wins == self.win_target:
            self.winner = "Good"
            return True  # Game Over

        current_player = self.pending_turns[self.current_action_type].pop(0)
        if current_player.is_agent():
            # Return so that the current team and everything can be updated
            return self.current_action_type

        if self.current_action_type == ActionType.TEAM_SELECTION:

            if self.current_proposal_number - 1 == self.max_proposals_allowed:
                return True  # Game over

            if not self.pending_turns[self.current_action_type]:
                raise Exception

            self.make_team_selection_move(current_player)

        elif self.current_action_type == ActionType.TEAM_APPROVAL:
            self.make_team_approval_move(current_player)

        elif self.current_action_type == ActionType.QUEST_VOTE:
            self.make_quest_vote_move(current_player)

        else:
            raise Exception("Invalid action type")

    def make_team_selection_move(self, player, override_choice=None):
        num_picks = self.quest_sizes[self.current_quest]
        if override_choice is not None:
            self.current_team = override_choice
        else:
            # Pick based on heuristics
            self.current_team = self.pick_quest_team(num_picks)

        self.current_leader = (self.current_leader + 1) % self.num_players
        self.current_proposal_number += 1
        self.initialize_team_approval_round()

    def make_team_approval_move(self, player, override_choice=None):
        response = override_choice
        if not response:
            response = player.approve_or_reject_quest()
        self.approvals += response

        if not self.pending_turns[ActionType.TEAM_APPROVAL]:
            self.conclude_team_approval_results()

    def make_quest_vote_move(self, player, override_choice=None):
        response = override_choice
        if not response:
            response = player.success_or_fail()

        if response is False:
            # Mission failed
            self.conclude_quest(False)

        if not self.pending_turns[ActionType.QUEST_VOTE]:
            # Mission succeeded
            self.conclude_quest(True)

    def conclude_team_approval_results(self):
        approved = 2 * self.approvals > self.num_players
        if approved:
            self.initialize_quest_vote_round()
        else:
            self.initialize_team_selection_round()

    def conclude_quest(self, success):
        self.current_quest += 1
        if success:
            self.good_team_wins += 1
        else:
            self.evil_team_wins += 1
        self.initialize_team_selection_round()

    def initialize_team_selection_round(self):
        self.current_team = None
        self.current_proposal_number = 0
        self.current_action_type = ActionType.TEAM_SELECTION
        self.pending_turns[ActionType.QUEST_VOTE] = self.generate_team_selection_turns()

    def initialize_team_approval_round(self):
        self.approvals = 0
        self.current_action_type = ActionType.TEAM_APPROVAL
        self.pending_turns[ActionType.TEAM_APPROVAL] = self.players[::]

    def initialize_quest_vote_round(self):
        self.current_action_type = ActionType.QUEST_VOTE
        self.pending_turns[ActionType.QUEST_VOTE] = self.current_team

    def pick_quest_team(self, num_picks):
        # For now, pick anyone except the leader
        # TODO: check
        # options = range(self.current_leader) + range(self.current_leader+1, self.num_players)
        options = range(self.num_players)
        player_ids = random.sample(options, num_picks)
        return [self.players[pid] for pid in player_ids]


game = AvalonGame(5)
agent = game.players[0]
feedback = game.run()

while feedback != True:
    if feedback == ActionType.TEAM_SELECTION:
        game.make_team_selection_move(agent)
    elif feedback == ActionType.TEAM_APPROVAL:
        game.make_team_approval_move(agent, True)
    elif feedback == ActionType.QUEST_VOTE:
        game.make_quest_vote_move(agent)
    feedback = game.run()
