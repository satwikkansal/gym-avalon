from enum import Enum


class CharacterType(Enum):
    """
    Different character types go here. And their behavior goes in the Quests class.
    """
    SERVANT = 1
    MERLIN = 2
    MINION = 3


class ActionType(Enum):
    """
    Different action types.
    """
    TEAM_SELECTION = 1
    TEAM_APPROVAL = 2
    QUEST_VOTE = 3


class Team(Enum):
    GOOD = 1
    EVIL = 2
