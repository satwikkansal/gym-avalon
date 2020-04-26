from enum import Enum
from itertools import product

MAX_QUESTS = 5


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
combinations = product(list(Team), [True, False], range(win_target), range(win_target))
for idx, combo in enumerate(combinations):
    PlayerVisibility[combo] = idx
