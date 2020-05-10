"""
This is the module that pieces together various components of the game (Quests, Players)
and serves as an interface to take actions in the game.
"""
import random

from game.enums_and_config import ActionType, CharacterType, Team, MAX_QUESTS, game_player_configs
from game.player import Player
from game.quest import Quest


class GameFeedback:
    """
    The glue between environment and the game. The AvalonGame.run function
    returns a GameFeedback object.

    It's a plain object with no functions.
    """
    def __init__(self, game, action_required=False, initiate_new_quest=False):
        quest = game.current_quest
        self.action_type = quest.current_action_type
        self.action_required = action_required
        self.game_winner = game.winner
        self.quest_winner = quest.quest_winner
        self.quest_number = quest.id
        self.quest_team_size = quest.team_size
        self.proposal_number = quest.current_proposal_number
        self.leader = quest.current_leader
        self.evil_wins = game.evil_team_wins
        self.initiate_new_quest = initiate_new_quest

        # self.current_team = [0] * game.num_players
        # for player in quest.current_team:
        #     self.current_team[player.player_id] = 1
        # self.current_team = tuple(self.current_team)
        self.current_team = quest.current_team


class AvalonGame:
    """
    The main game class.
    """
    def __init__(self, num_players, max_quests=MAX_QUESTS, max_proposals_allowed=5, enable_logs=True, majority_rule=True):
        """
        Setting up logic for the game goes here. Number of players are configurable.
        """
        self.enable_logs = enable_logs
        self.majority_rule = majority_rule
        self.num_players = num_players
        self.players = self.initialize_players()

        self.max_quests = max_quests
        self.max_proposals_allowed = max_proposals_allowed
        # Number of quests needed to be won by a team to win the game
        self.win_target = max_quests // 2 + 1
        self.quest_sizes = game_player_configs[num_players].quest_size

        self.current_quest = None
        self.current_player = None
        self.initialize_new_quest()

        self.good_team_wins = 0
        self.evil_team_wins = 0

        self.winner = None

    def initialize_new_quest(self):
        """
        This function should be called before starting any new quest in the game.
        """
        quest_num = 0
        leader = 0
        if self.current_quest is not None:
            quest_num = self.current_quest.id + 1
            leader = self.current_quest.current_leader

        self.current_quest = Quest(quest_num=quest_num,
                                   team_size=self.quest_sizes[quest_num],
                                   players=self.players,
                                   leader=leader,
                                   enable_logs=self.enable_logs,
                                   majority_rule=self.majority_rule)
        self.current_player = self.current_quest.current_leader

    def __str__(self):
        """
        Method for displaying nice print(game) summaries.
        """
        string = f"""
        =======================
        Wins: Good {self.good_team_wins} / Evil {self.evil_team_wins}
        Current Quest --> {self.current_quest}
        """
        return string

    def initialize_players(self):
        """
        Assigns player types based on the number of players and game_player_configs.
        """
        # Random sampling for random assignment of character types
        player_order = random.sample(range(self.num_players), self.num_players)
        good_count, evil_count = game_player_configs[self.num_players].team_split

        assert good_count + evil_count == self.num_players

        players = []
        for player_id, order in enumerate(player_order):
            if order + 1 <= good_count:
                player = Player(player_id, CharacterType.SERVANT)
            else:
                player = Player(player_id, CharacterType.MINION)
            players.append(player)

        return players

    def run(self, agent_player=None, override_choice=None):
        """
        The driving logic of the game.

        The game step runs on its own unless the current player is an agent.
        In that case the run method will return a feedback indicating a desired action,
        and the next step is to call the run method again by passing `agent_player` and
        `override_choice` arguments.
        """
        quest = self.current_quest

        if agent_player:
            assert self.current_player == agent_player
        else:
            self.current_player = quest.get_next_player()
            if self.current_player.is_agent() and agent_player is None:
                # Return so that the current team and everything can be updated
                return GameFeedback(self, action_required=True)

        if quest.current_action_type == ActionType.TEAM_SELECTION:
            if quest.current_proposal_number - 1 == self.max_proposals_allowed:
                self.winner = Team.EVIL
            else:
                quest.make_team_selection_move(self.current_player, override_choice=override_choice)
            return GameFeedback(self)

        elif quest.current_action_type == ActionType.TEAM_APPROVAL:
            quest.make_team_approval_move(self.current_player, override_choice=override_choice)
            return GameFeedback(self)

        elif quest.current_action_type == ActionType.QUEST_VOTE:
            winner = quest.make_quest_vote_move(self.current_player, override_choice=override_choice)
            if winner:
                self.update_scores(winner)
                # self.initialize_new_quest()
                return GameFeedback(self, initiate_new_quest=True)

            return GameFeedback(self)

        raise Exception(f"Game entered in some unknown path for action type {quest.current_action_type}")

    def update_scores(self, winner):
        self.good_team_wins += winner == Team.GOOD
        self.evil_team_wins += winner == Team.EVIL
        if self.good_team_wins == self.win_target:
            self.winner = Team.GOOD
        elif self.evil_team_wins == self.win_target:
            self.winner = Team.EVIL
