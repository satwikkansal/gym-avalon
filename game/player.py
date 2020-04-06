import random

from game.enums import CharacterType, Team

auto_player_configs = {

}


class Player:
    def __init__(self, player_id, char_type):
        self.player_id = player_id
        self.char_type = char_type
        if char_type == CharacterType.MINION:
            self.team = Team.EVIL
        else:
            self.team = Team.GOOD

    def __str__(self):
        return f'Player {self.player_id} ({self.char_type})'

    # Action
    def approve_or_reject_quest_team(self):
        """
        Heuristics based function to decide whether the player should approve
        or reject the team.

        Some heuristics incorporated are,

        Good orientation player,
        - More likely to reject a person who has been a part of a failed quest before.
        - More likely to approve is the team involves himself (as he knows he has good orientation)

        Bad orientation player,
        - More likely to approve if the team involves a team-mate who is known to be evil.
        - More likely to approve if the team involves himself (as he know he has bad orientation and can sabotage)
        - More likely to reject if the team consists of just the good players.

        Store the quest history in the form,

        [{'winner'}]

        :return: True or False depending on the decision taken based on the heuristics.
        """
        return True

    def success_or_fail(self):
        """
        Heuristics based function to decide whether the player should pass or fail the quest.

        Some heuristics used are,

        For Good orientation,
        - Always pass the mission

        For Bad orientation
        - More likely to fail the mission.

        :return:  True or False depending on the decision taken based on the heuristics.
        """
        if self.team == Team.EVIL:
            # TODO: Add heuristics here, probability
            return False
        else:
            return True

    def pick_quest_team(self, quest):
        """
        Heuristics based function to decide whether the player should pass or fail the quest.

        Some heuristics used are

        For Good orientation,
        - More likely to pick itself or those who have been part of a successful mission already.

        For Bad orientation
        - More likely to pick itself any of its team members.

        :return:  True or False depending on the decision taken based on the heuristics.
        """
        options = range(quest.num_players)
        player_ids = random.sample(options, quest.num_players)
        return [quest.quest_players[pid] for pid in player_ids]

    def is_agent(self):
        return self.player_id == 0

    @staticmethod
    def players_string(players_list):
        return ', '.join([str(p) for p in players_list])