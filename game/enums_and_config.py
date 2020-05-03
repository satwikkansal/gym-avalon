from collections import namedtuple
from enum import Enum
from itertools import product, combinations


MAX_QUESTS = 5


player_config = namedtuple('player_config', ['team_split', 'quest_size', 'max_team_selection_options'])


game_player_configs = {
    5: player_config([3, 2], [2, 3, 2, 3, 3], 10),  # 5-choose-2
    6: player_config([4, 2], [2, 3, 4, 3, 4], 20),  # 6-choose-3
    7: player_config([4, 3], [2, 3, 3, 4, 4], 35),
    8: player_config([5, 3], [3, 4, 4, 5, 5], 70),
    9: player_config([6, 3], [3, 4, 4, 5, 5], 126),
    10: player_config([6, 4], [3, 4, 4, 5, 5], 252),
    11: player_config([7, 4], [4, 5, 4, 5, 5], 462)
}


class CharacterType(Enum):
    """
    Different character types go here. And their behavior goes in the Quests class.
    """
    SERVANT = 1
    MINION = 2


class ActionType(Enum):
    """
    Different action types.
    """
    TEAM_SELECTION = 0
    TEAM_APPROVAL = 1
    QUEST_VOTE = 2


class Team(Enum):
    GOOD = 1
    EVIL = 2
    UNKNOWN = 3


PlayerVisibility = {}
win_target = MAX_QUESTS // 2 + 2

# Which, Part of current team or not, Failed , Passed mission
combos = product(list(Team), [True, False], range(win_target), range(win_target))
for idx, combo in enumerate(combos):
    PlayerVisibility[combo] = idx


def generate_combinations(n, c):
    """
    Generate all possible ways to select c numbers out of n given ones.
    """
    combos = []
    for combination in combinations(range(n), c):
        combos.append(list(combination))
    return combos


def generate_team_selection_combinations(n_players, config):
    """
    Assigns a scalar value to all possible n-choose-p type player type combinations.
    :param n_players: total number of players available
    :param config: the player_config
    :return:
    """
    all_combos = []
    seen = set()
    for size in config.quest_size:
        if size not in seen:
            seen.add(size)
            all_combos += generate_combinations(n_players, size)
    return all_combos

team_selection_move_map = {}

# Pre-load the team_selection_maps for all the game configurations available.
for n_players, config in game_player_configs.items():
    team_selection_move_map[n_players] = generate_team_selection_combinations(n_players, config)
