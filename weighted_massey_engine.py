import numpy as np
import pandas as pd
from scipy import stats


class WeightedMasseyEngine:
    @classmethod
    def get_ratings(cls, goals_home, goals_away, teams_home,
        teams_away, match_date = None, avg_mv = None) -> pd.DataFrame:
        """
        Gets the Weighted Massey ratings

        Returns
        -------
        ratings: 
            a dataframe containing  
                1) rating: weighted massey rating per team (weighted by date and homefield advantage)
                2) mv_rating: weighted massey in combination with transformed transfer market value
        
        home_advantage: 
            automatically calculated homefield advantage rating
        """

        teams = np.sort(np.unique(np.concatenate([teams_home, teams_away])))

        fixtures = pd.DataFrame(
            [goals_home, goals_away, teams_home, teams_away]
        ).T
        fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
        fixtures["goals_home"] = fixtures["goals_home"].astype(int)
        fixtures["goals_away"] = fixtures["goals_away"].astype(int)
        
        # Weighted Time
        if match_date is not None:
            match_date = pd.to_datetime(match_date)
            # Normalize the match date to a range between 0 and 1
            min_time = match_date.min()
            max_time = match_date.max()
            # Apply an exponential function to the normalized time
            fixtures["weighted_time"] = np.exp((match_date - min_time) / (max_time - min_time)) / np.exp(1) # Dividing by np.exp(1) to ensure the values are scaled between 0 and 1
        else:
            fixtures["weighted_time"] = 1

        fixtures["wtd_goals_home"] = fixtures["goals_home"] * fixtures["weighted_time"]
        fixtures["wtd_goals_away"] = fixtures["goals_away"] * fixtures["weighted_time"]

        # Build the linear system and solve it
        ratings = cls._solve_ratings(fixtures, teams)
        r = ratings[:-1]
        home_advantage = ratings[-1]

        res = pd.DataFrame([teams, r]).T
        res.columns = ["team", "rating"]
        res["team"] = res["team"].astype(int)
            
        res = res.sort_values("rating", ascending=False)
        res = res.reset_index(drop=True)

        # Average TransferMarket Value
        if avg_mv is not None:
            # apply box-cox transformation to normalize average market value, then standardize it to range (0,1)
            avg_mv = avg_mv.copy()
            transformed_mv, _ = stats.boxcox(avg_mv["avg_market_val"])
            #transformed_mv = transformed_mv.copy()
            avg_mv["transformed_mv"] = (transformed_mv - transformed_mv.min()) / (transformed_mv.max() - transformed_mv.min())
            res = res.merge(avg_mv, left_on="team",right_on="team_id")
            res["mv_rating"] = res["rating"] + res["transformed_mv"]
            columns_to_keep = ["team", "rating", "mv_rating", "season"]
            res = res[columns_to_keep]

        return res, home_advantage

    # in matrix M, 1 = game played at home, -1 = game played away
    @classmethod
    def _solve_ratings(cls, fixtures, teams):
        N = len(teams)
        M = np.zeros((N+1, N+1))
        p = np.zeros(N+1)
        team_to_matrix_idx = {t:i for i,t in enumerate(teams)}

        for team_home,team_away,weighted_time,wtd_goals_home,wtd_goals_away in fixtures[["team_home","team_away","weighted_time","wtd_goals_home","wtd_goals_away"]].itertuples(False):
            h = team_to_matrix_idx[team_home]
            a = team_to_matrix_idx[team_away]
            M[h, a] -= 1 * weighted_time
            M[a, h] -= 1 * weighted_time
            M[h, -1] += 1
            M[a, -1] -= 1
            wtd_goal_diff = wtd_goals_home - wtd_goals_away
            p[h] += wtd_goal_diff
            p[a] -= wtd_goal_diff
    
        d_idxs = np.arange(N)
        M[d_idxs,d_idxs] = np.abs(np.sum(M[:N,:N],axis=1))
        # fill in the last row
        M[-1,:] = M[:,-1]
        M[-1,-1] = len(fixtures)

        # append the homefiled advantage term
        p[-1] = np.sum(np.abs(fixtures["wtd_goals_home"] - fixtures["wtd_goals_away"]))   

        ratings = np.linalg.lstsq(M, p, rcond=None)[0]
        return ratings
