import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

class Colley:
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
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        match_date,
        include_draws=True,
        draw_weight=0.5,
    ):
        self.goals_home = goals_home
        self.goals_away = goals_away
        self.teams_home = teams_home
        self.teams_away = teams_away
        self.include_draws = include_draws
        self.match_date = match_date
        self.draw_weight = draw_weight

    def get_ratings(self) -> pd.DataFrame:
        """
        Gets the Colley ratings

        Returns
        -------
            Returns a dataframe containing colley ratings per team
        """
        teams = np.sort(np.unique(np.concatenate([self.teams_home, self.teams_away])))

        fixtures = _build_fixtures(
            self.goals_home, self.goals_away, self.teams_home, self.teams_away
        )

        # Weighted Time
        if self.match_date is not None:
            fixtures['match_date'] = pd.to_datetime(self.match_date)
            # Normalize the match date to a range between 0 and 1
            min_time = fixtures['match_date'].min()
            max_time = fixtures['match_date'].max()
            fixtures['time_normalized'] = (fixtures['match_date'] - min_time) / (max_time - min_time)
            # Apply an exponential function to the normalized time
            fixtures['weighted_time'] = np.exp(fixtures['time_normalized']) / np.exp(1)  # Dividing by np.exp(1) to ensure the values are scaled between 0 and 1

        else:
            fixtures['weighted_time'] = 1

        fixtures["game_weight"] = fixtures['weighted_time']

        C, b = _build_C_b(fixtures, teams, self.include_draws, self.draw_weight)

        r = _solve_r(C, b)
        r = pd.DataFrame([teams, r]).T
        r.columns = ["team", "rating"]
        r = r.sort_values("rating", ascending=False)
        r = r.reset_index(drop=True)

        return r


def _build_fixtures(goals_home, goals_away, teams_home, teams_away):
    fixtures = pd.DataFrame([goals_home, goals_away, teams_home, teams_away]).T
    fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
    fixtures["goals_home"] = fixtures["goals_home"].astype(int)
    fixtures["goals_away"] = fixtures["goals_away"].astype(int)
    return fixtures


def _solve_r(C, b):
    r = np.linalg.solve(C, b)
    return r


def _build_C_b(fixtures, teams, include_draws, draw_weight):
    n_teams = len(teams)
    C = np.zeros([n_teams, n_teams])
    b = np.zeros([n_teams])

    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        C[h, a] = C[h, a] - row["game_weight"]
        C[a, h] = C[a, h] - row["game_weight"]

        if row["goals_home"] > row["goals_away"]:
            b[h] += row["game_weight"]
            b[a] -= row["game_weight"]

        elif row["goals_home"] < row["goals_away"]:
            b[h] -= row["game_weight"]
            b[a] += row["game_weight"]

        else:
            if include_draws:
                b[h] += draw_weight
                b[a] += draw_weight

    np.fill_diagonal(C, np.abs(C.sum(axis=1)) + 2)
    b = 1 + b * 0.5

    return C, b


con = sqlite3.connect("english_football_data.sqlite")

gamesQuery = f"""
SELECT home_team_id, away_team_id, fulltime_home_goals, fulltime_away_goals, fulltime_result, date
FROM Matches
WHERE season != {2023}
AND league_id = {1}
"""

Games = pd.read_sql_query(gamesQuery, con)

home_goals_list = Games["fulltime_home_goals"]
away_goals_list = Games["fulltime_away_goals"]
home_teams_list = Games['home_team_id']
away_teams_list = Games['away_team_id']
match_dates_list = Games['date']

colley = Colley(home_goals_list,away_goals_list,home_teams_list, away_teams_list,match_dates_list)

colley_ratings = colley.get_ratings()


print(colley_ratings)

