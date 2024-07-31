import numpy as np
import pandas as pd


class WeightedColleyEngine:
    @classmethod
    def get_ratings(cls, goals_home, goals_away, teams_home, teams_away,
                    match_date=None, include_draws=True, draw_weight=0.5) -> pd.DataFrame:
        """
        Calculates each team's Colley ratings

        Parameters
        ----------
        goals_home : list
            List of goals scored by the home teams

        goals_away : list
            List of goals scored by the away teams

        teams_home : list
            List of names of the home teams

        teams_away : list
            List of names of the away teams

        include_draws : bool
            Should tied results be included in the ratings?

        draw_weight : float
            if include_draws is `True` then this sets the weighting applied to tied scores.
            For example `0.5` means a draw is worth half a win, `0.333` means a draw
            is a third of a win etc

        Returns
        -------
            Returns a dataframe containing colley ratings per team
        """
        teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))

        fixtures = cls._build_fixtures(
            goals_home, goals_away, teams_home, teams_away
        )

        # Weighted Time
        if match_date is not None:
            match_date = pd.to_datetime(match_date)
            # Normalize the match date to a range between 0 and 1
            min_time = match_date.min()
            max_time = match_date.max()
            # Apply an exponential function to the normalized time
            fixtures['weighted_time'] = np.exp((match_date - min_time) / (max_time - min_time)) / np.exp(1)  # Dividing by np.exp(1) to ensure the values are scaled between 0 and 1
        else:
            fixtures['weighted_time'] = 1
        fixtures["game_weight"] = fixtures['weighted_time']

        r = cls._solve_ratings(fixtures, teams, include_draws, draw_weight)
        r = pd.DataFrame([teams, r]).T
        r.columns = ["team", "rating"]
        r = r.sort_values("rating", ascending=False)
        r = r.reset_index(drop=True)

        return r

    @classmethod
    def _build_fixtures(cls, goals_home, goals_away, teams_home, teams_away):
        fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
        fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
        fixtures["goals_home"] = fixtures["goals_home"].astype(int)
        fixtures["goals_away"] = fixtures["goals_away"].astype(int)
        return fixtures

    @classmethod
    def _solve_ratings(cls, fixtures, teams, include_draws, draw_weight):
        N = len(teams)
        C = np.zeros((N,N))
        b = np.zeros(N)
        team_to_matrix_idx = {t:i for i,t in enumerate(teams)}

        for goals_home,goals_away,team_home,team_away,game_weight in fixtures[["goals_home","goals_away","team_home","team_away","game_weight"]].itertuples(False):
            h = team_to_matrix_idx[team_home]
            a = team_to_matrix_idx[team_away]

            C[h, a] -= game_weight
            C[a, h] -= game_weight

            result = np.sign(goals_home - goals_away)
            if result:
                b[h] += game_weight * result
                b[a] -= game_weight * result
            elif include_draws:
                b[h] += game_weight * draw_weight
                b[a] += game_weight * draw_weight

        np.fill_diagonal(C, np.abs(C.sum(axis=1)) + 2)
        b = 1 + b * 0.5

        r = np.linalg.solve(C, b)
        return r
