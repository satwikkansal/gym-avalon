from game.enums import ActionType, Team
from game.player import  Player


class Quest:
    def __init__(self, quest_num, team_size, players, leader):
        self.id = quest_num
        self.team_size = team_size
        self.quest_players = players
        self.num_players = len(players)
        self.current_leader = leader
        self.current_proposal_number = 0
        self.current_team = []
        self.approvals = 0

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
            Approvals so far: {self.approvals}
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
        self.current_proposal_number = 0
        self.current_action_type = ActionType.TEAM_SELECTION
        self.pending_turns = dict()
        self.pending_turns[ActionType.TEAM_SELECTION] = self.generate_team_selection_turns()

    def initialize_team_approval_round(self):
        """
        Method to initialize team approval round.
        """
        self.approvals = 0
        self.current_action_type = ActionType.TEAM_APPROVAL
        self.pending_turns = dict()
        self.pending_turns[ActionType.TEAM_APPROVAL] = self.quest_players[::]

    def initialize_quest_vote_round(self):
        """
        Method to initialize quest voting round.
        """
        self.current_action_type = ActionType.QUEST_VOTE
        self.pending_turns = dict()
        self.pending_turns[ActionType.QUEST_VOTE] = self.current_team

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
            self.current_team = override_choice
        else:
            self.current_team = player.pick_quest_team(self)

        print(f"{player} picked the team {Player.players_string(self.current_team)}")

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
        self.approvals += response

        print(f'{player} {"Approved" if response else "Rejected"} the team')

        if not self.pending_turns[ActionType.TEAM_APPROVAL]:
            self.conclude_team_approval_results()

    def conclude_team_approval_results(self):
        """
        Helper method to decide if the proposed team is approved or not.
        """
        approved = 2 * self.approvals > self.num_players
        if approved:
            print("Current team is approved!")
            self.initialize_quest_vote_round()
        else:
            print("Team approval failed")
            self.initialize_team_selection_round()

    def make_quest_vote_move(self, player, override_choice=None):
        """
        Quest voting move with a provision for providing override by the agent.
        """
        response = override_choice
        if not response:
            response = player.success_or_fail()

        print(f'{player} {"Passed" if response else "Failed"} the quest')

        if response is False:
            # Mission failed
            print(f'Mission Failed due to {player}')
            return self.conclude_quest(False)

        if not self.pending_turns[ActionType.QUEST_VOTE]:
            # Mission succeeded
            print(f'Mission Succeeded!')
            return self.conclude_quest(True)

    def conclude_quest(self, success):
        if success:
            print("Good team won the quest!")
            self.quest_winner = Team.GOOD
        else:
            print("Evil team won the quest!")
            self.quest_winner = Team.EVIL
        return self.quest_winner
