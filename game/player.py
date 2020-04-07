import numpy as np

from game.enums import CharacterType, Team


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
    def approve_or_reject_quest_team(self, quest):
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

        :return: True or False depending on the decision taken based on the heuristics.
        """
        approve_self_bias = 0.8
        evil_approve_bias = 0.9

        # Irrespective of the team
        if self in quest.current_team:
            p = approve_self_bias
        else:
            p = (1 - approve_self_bias)

        if self.team is Team.EVIL:
            for player in quest.current_team:
                if player.team is Team.EVIL:
                    p = evil_approve_bias
                    break

        return self.decide_with_probability(p)

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
        p = 1.0
        evil_mission_fail_prob = 0.8
        if self.team is Team.EVIL:
           p = 1 - evil_mission_fail_prob

        return self.decide_with_probability(p)

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
        weights = [10] * quest.num_players

        if self.team is Team.EVIL:
            for pid, player in enumerate(quest.quest_players):
                if player.team is Team.EVIL:
                    weights[pid] = 20

        weights[self.player_id] = 20
        player_ids = self.weighted_exclusive_choice(options, quest.num_players, weights)
        return [quest.quest_players[pid] for pid in player_ids]

    @staticmethod
    def weighted_exclusive_choice(options, num_picks, weights):
        weights_sum = sum(weights)
        if weights_sum != 1.0:
            # Normalize the weights
            weights = [w / weights_sum for w in weights]
        return np.random.choice(options, num_picks, False, weights)

    @staticmethod
    def decide_with_probability(p):
        return np.random.binomial(1, p) == 1

    def is_agent(self):
        return self.player_id == 0

    @staticmethod
    def players_string(players_list):
        return ', '.join([str(p) for p in players_list])