from enum import Enum


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
