from collections import namedtuple
from enum import Enum

import math
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


class Player:
    def __init__(self, player_id, char_type):
        self.player_id = player_id
        self.char_type = char_type

    # Action
    def approve_or_reject_quest(self):
        return True

    def is_evil(self):
        """
        :return: True or False depending upon the character type
        """
        return self.char_type is CharacterType.MINION

    def success_or_fail(self):
        if self.is_evil():
            # TODO: Add heuristics here
            return False
        else:
            return True


class AvalonGame:
    def __init__(self, num_players, max_quests=5):
        self.num_players = num_players
        self.players = self.initalize_players()
        self.current_quest = 0
        self.max_quests = max_quests
        # Number of quests needed to be won by a team to win the game
        self.win_target = max_quests // 2 + 1
        self.quest_sizes = game_player_configs[num_players].quest_size

        self.current_leader = 0

    def initalize_players(self):
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

    def start_quest(self):
        num_picks = self.quest_sizes[self.current_quest]
        for _ in range(self.num_players):
            # Action
            team = self.pick_quest_team(num_picks)
            self.current_leader += 1
            if self.check_team_approval(team):
                nonlocal quest_pass
                # Action
                quest_pass = self.execute_mission(team)
                break
        else:
            print("No one agreed upon a mission team")
            quest_pass = False

        self.current_quest += 1
        return quest_pass

    def pick_quest_team(self, num_picks):
        # For now, pick anyone except the leader
        # TODO: check
        # options = range(self.current_leader) + range(self.current_leader+1, self.num_players)
        options = range(self.num_players)
        player_ids = random.sample(options, num_picks)
        return [self.players[pid] for pid in player_ids]

    def check_team_approval(self, team):
        team_size = len(team)
        approvals = 0
        for player in team:
            approvals += player.approve_or_reject_quest()

        return approvals * 2 > team_size

    def execute_mission(self, team: list[Player]):
        """
        Returns False if any of the team player sabotages the mission.
        """
        for player in team:
            if not player.success_or_fail():
                return False
        return True


