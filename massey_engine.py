import numpy as np
import pandas as pd


class MasseyEngine:
    @classmethod
    def get_ratings(cls, goals_home, goals_away, teams_home, teams_away) -> pd.DataFrame:
        teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))

        fixtures = pd.DataFrame(
            [goals_home, goals_away, teams_home, teams_away]
        ).T
        fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
        fixtures["goals_home"] = fixtures["goals_home"].astype(int)
        fixtures["goals_away"] = fixtures["goals_away"].astype(int)

        r = cls._solve_ratings(fixtures, teams)

        res = pd.DataFrame([teams, r]).T
        res.columns = ["team", "rating"]
        res = res.sort_values("rating", ascending=False)
        res = res.reset_index(drop=True)
        return res

    @classmethod
    def _solve_ratings(cls, fixtures, teams):
        N = len(teams)
        M = np.zeros((N+1, N))
        p = np.zeros(N+1)
        team_to_matrix_idx = {t:i for i,t in enumerate(teams)}

        for goals_home,goals_away,team_home,team_away in fixtures.itertuples(False):
            h = team_to_matrix_idx[team_home]
            a = team_to_matrix_idx[team_away]
            M[h, a] -= 1
            M[a, h] -= 1
            goal_diff = goals_home - goals_away
            p[h] += goal_diff
            p[a] -= goal_diff

        d_idxs = np.arange(N)
        M[d_idxs,d_idxs] = np.abs(np.sum(M,axis=1))[:-1]
        M[N,:] = 1

        ratings = np.linalg.lstsq(M, p, rcond=None)[0]
        return ratings
