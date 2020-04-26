from game.enums import ActionType, Team
from game.player import  Player


class Quest:
    def __init__(self, quest_num, team_size, players, leader, enable_logs):
        self.enable_logs = enable_logs
        self.id = quest_num
        self.team_size = team_size
        self.quest_players = players
        self.num_players = len(players)
        self.current_leader = leader
        self.current_proposal_number = 0
        self.current_team = []
        self.quest_team_approvals = 0
        self.mission_fails = 0

        # Info for keeping track of whose turn is next.
        self.current_action_type = ActionType.TEAM_SELECTION
        self.pending_turns = {
            ActionType.TEAM_SELECTION: self.generate_team_selection_turns()
        }

        self.quest_winner = None

    def __str__(self):
        quest_string = f"""
        Quest number: {self.id}
        Current Action: {self.current_action_type}
        Leader: {self.current_leader}
        Current team: {Player.players_string(self.current_team)}
        """

        if self.current_action_type == ActionType.TEAM_APPROVAL:
            team_approval_info = f'''
            Current Proposal: {self.current_proposal_number}
            Approvals so far: {self.quest_team_approvals}
            '''
            quest_string += team_approval_info

        # Prints the player pending turns in a readable format.
        player_turns_string = "\n\t\t".join(
            [f"{k}: {Player.players_string(v)}" for k, v in self.pending_turns.items()])
        player_turns_info = f'''
        Next turns: 
        {player_turns_string}
        '''
        return quest_string + player_turns_info

    def get_next_player(self):
        return self.pending_turns[self.current_action_type].pop(0)

    """
    Following are initialization functions for different rounds.
    """
    def initialize_team_selection_round(self):
        """
        Method to initialize team selection round.
        """
        self.current_team = []
        self.current_action_type = ActionType.TEAM_SELECTION
        self.pending_turns = dict()
        self.pending_turns[ActionType.TEAM_SELECTION] = self.generate_team_selection_turns()

    def initialize_team_approval_round(self):
        """
        Method to initialize team approval round.
        """
        self.quest_team_approvals = 0
        self.current_action_type = ActionType.TEAM_APPROVAL
        self.pending_turns = dict()
        self.pending_turns[ActionType.TEAM_APPROVAL] = self.quest_players[::]

    def initialize_quest_vote_round(self):
        """
        Method to initialize quest voting round.
        """
        self.current_action_type = ActionType.QUEST_VOTE
        self.pending_turns = dict()
        self.pending_turns[ActionType.QUEST_VOTE] = self.current_team[:]

    def generate_team_selection_turns(self):
        """
        Starts from current leader, and generate turns in the circular fashion.
        """
        return [self.quest_players[(self.current_leader + i) % self.num_players] for i in
                range(len(self.quest_players))]

    """
    Following are the move methods for every round. Any move related logic should be added here 
    or in the Player class method's that these functions call.
    """
    def make_team_selection_move(self, player, override_choice=None):
        """
        Team selection move with a provision for providing override by the agent.
        """
        if override_choice is not None:
            assert len(override_choice) == self.team_size
            player_ids = override_choice
        else:
            player_ids = player.pick_quest_team(self)

        self.current_team = [self.quest_players[pid] for pid in player_ids]

        if self.enable_logs: print(f"{player} picked the team {Player.players_string(self.current_team)}")

        self.current_leader = (self.current_leader + 1) % self.num_players
        self.current_proposal_number += 1
        self.initialize_team_approval_round()

    def make_team_approval_move(self, player, override_choice=None):
        """
        Team approval move with a provision for providing override by the agent.
        """
        response = override_choice
        if not response:
            response = player.approve_or_reject_quest_team(self)
        self.quest_team_approvals += response

        if self.enable_logs: print(f'{player} {"Approved" if response else "Rejected"} the team')

        if not self.pending_turns[ActionType.TEAM_APPROVAL]:
            self.conclude_team_approval_results()

    def conclude_team_approval_results(self):
        """
        Helper method to decide if the proposed team is approved or not.
        """
        approved = 2 * self.quest_team_approvals > self.num_players
        if approved:
            if self.enable_logs: print("Current team is approved!")
            self.initialize_quest_vote_round()
        else:
            if self.enable_logs: print("Team approval failed")
            self.initialize_team_selection_round()

    def make_quest_vote_move(self, player, override_choice=None):
        """
        Quest voting move with a provision for providing override by the agent.
        """
        response = override_choice
        if not response:
            response = player.success_or_fail()

        if self.enable_logs: print(f'{player} {"Passed" if response else "Failed"} the quest')

        self.mission_fails += response==False

        # The following comments are for the Previous version where a single
        # fail was enough to sabotage the mission.

        # if response is False:
        #     # Mission failed
        #     if self.enable_logs: print(f'Mission Failed due to {player}')
        #     return self.conclude_quest(False)

        if not self.pending_turns[ActionType.QUEST_VOTE]:
            # if self.enable_logs: print(f'Mission Succeeded!')
            # return self.conclude_quest(True)
            return self.conclude_quest_majority()

    """
    Previous version method of concluding quest where a single 
    fail was enough to sabotage the mission.
    def conclude_quest(self, success):
        if success:
            if self.enable_logs: print("Good team won the quest!")
            self.quest_winner = Team.GOOD
        else:
            if self.enable_logs: print("Evil team won the quest!")
            self.quest_winner = Team.EVIL
        return self.quest_winner
    """

    def conclude_quest_majority(self):
        # If half or more team members fail mission, then only it's a fail
        success = 2 * self.mission_fails < self.team_size
        if success:
            if self.enable_logs: print("Good team won the quest!")
            self.quest_winner = Team.GOOD
        else:
            if self.enable_logs: print("Evil team won the quest!")
            self.quest_winner = Team.EVIL

        self.mission_fails = 0
        self._update_player_histories()
        return self.quest_winner

    def _update_player_histories(self):
        """
        To keep track of the history of passed or failed mission for a player.
        """
        for player in self.current_team:
            player.mission_history.append(self.quest_winner is Team.GOOD)
