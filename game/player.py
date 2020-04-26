import numpy as np

from game.enums import CharacterType, Team

"""
Good player strategies,

Source: https://www.quora.com/What-are-some-rookie-mistakes-in-the-Avalon-game

Rookie mistakes
- Good players overuse approval vote: if you are good and don't have a special role, probabilistically speaking only up to 2 out of the 7 proposals 
  (including your own) you hear around the table do not have any bad guys sneaked in. f generic good players are too happy to approve proposed missions, 
  then they'll send too many bad teams on quests and lose the game by default.
"""


class Player:
    def __init__(self, player_id, char_type):
        self.player_id = player_id
        self.char_type = char_type
        if char_type == CharacterType.MINION:
            self.team = Team.EVIL
        else:
            self.team = Team.GOOD

        self.mission_history = []

    def __str__(self):
        return f'Player {self.player_id} ({self.char_type} {self.mission_history})'

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
        good_self_approve_bias = 0.9
        good_approve_bias = 0.8
        suspicion_factor = 0.2
        evil_approve_bias = 0.9

        majority_size = (quest.team_size + 1) // 2

        if self.team is Team.EVIL:
            p = 0.0
            evils_in_team = 0
            for player in quest.current_team:
                if player.team is Team.EVIL:
                    evils_in_team += 1
            if evils_in_team >= majority_size:
                p = evil_approve_bias
            else:
                p = (1 - evil_approve_bias)

        if self.team is Team.GOOD:
            if self in quest.current_team:
                p = good_self_approve_bias
            else:
                p = good_approve_bias

            for player in quest.current_team:
                if False in player.mission_history:
                    p -= suspicion_factor

            p = max([suspicion_factor, p])

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
        evil_mission_fail_prob = 0.9
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
        weights = [100] * quest.num_players

        if self.team is Team.EVIL:
            for pid, player in enumerate(quest.quest_players):
                if player.team is Team.EVIL:
                    weights[pid] += 200 # More weight to evil players

        if self.team is Team.GOOD:
            for pid, player in enumerate(quest.quest_players):
                # The good player doesn't know who is good and evil, so use history
                for mission_success in player.mission_history:
                    if mission_success:
                        weights[pid] += 50
                    else:
                        weights[pid] -= 50

            weights[self.player_id] = 500

        player_ids = self.weighted_exclusive_choice(options, quest.team_size, weights)
        return player_ids

    @staticmethod
    def weighted_exclusive_choice(options, num_picks, weights):
        """
        Make exclusive num_choices based on the weights provided.
        """
        weights = [w if w > 0 else 0.1 for w in weights]
        weights_sum = sum(weights)
        if weights_sum != 1.0:
            # Normalize the weights
            weights = [w / weights_sum for w in weights]
        return np.random.choice(options, num_picks, False, weights)

    @staticmethod
    def decide_with_probability(p):
        """
        Bernoulli's distribution
        """
        return np.random.binomial(1, p) == 1

    def is_agent(self):
        # For now, player with ID 0 is the agent.
        return self.player_id == 0

    @staticmethod
    def players_string(players_list):
        return ', '.join([str(p) for p in players_list])