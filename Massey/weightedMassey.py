import numpy as np
import pandas as pd
from scipy import stats


class WeightedMassey:
    """
    Calculates each team's Massey ratings with weighted match date and inclusion of homefield advantage

    Input
    ----------
    goals_home : list
        List of goals scored by the home teams

    goals_away : list
        List of goals scored by the away teams

    teams_home : list
        List of names of the home teams

    teams_away : list
        List of names of the away teams
    
    match_date : list
        List of dates of which the game is played
    
    avg_mv: DataFrame
        A DataFrame with the target season's average market value per team and  team id as index
    """

    def __init__(
        self,
        goals_home,
        goals_away,
        teams_home,
        teams_away,
        match_date =  None,
        avg_mv = None
    ):
        self.goals_home = goals_home
        self.goals_away = goals_away
        self.teams_home = teams_home
        self.teams_away = teams_away
        self.match_date = match_date
        self.avg_mv = avg_mv

    def get_ratings(self) -> pd.DataFrame:
        """
        Gets the Weighted Massey ratings

        Returns
        -------
        ratings: 
            a dataframe containing  
                1) rating: weighted massey rating per team (weighted by date and homefield advantage)
                2) offence: offence score per team
                3) defence: defence score per team
                4) mv_rating: weighted massey in combination with transformed transfer market value
        
        home_advantage: 
            automatically calculated homefield advantage rating
        """

        teams = np.sort(np.unique(np.concatenate([self.teams_home, self.teams_away])))

        fixtures = pd.DataFrame(
            [self.goals_home, self.goals_away, self.teams_home, self.teams_away]
        ).T
        fixtures.columns = ["goals_home", "goals_away", "team_home", "team_away"]
        fixtures["goals_home"] = fixtures["goals_home"].astype(int)
        fixtures["goals_away"] = fixtures["goals_away"].astype(int)

        
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

        fixtures["wtd_goals_home"] = fixtures['goals_home'] * fixtures['weighted_time']
        fixtures['wtd_goals_away'] = fixtures['goals_away'] * fixtures['weighted_time']



        # Build the linear system and solve it
        M = _build_m(fixtures, teams)
        p = _build_p(fixtures, teams)
        ratings = _solve_ratings(M, p)
        r = ratings[:-1]
        home_advantage = ratings[-1]

        t = _build_t(fixtures, teams)
        f = _build_f(fixtures, teams)
        Tr_f = (np.diag(t) * ratings) - f
        d = _solve_d(t, Tr_f)[:-1]
        o = r - d

        res = pd.DataFrame([teams, r, o, d]).T
        res.columns = ["team", "rating", "offence", "defence"]
        if res['team'].dtype == 'float64':
            res['team'] = res['team'].astype('int')
            
        res = res.sort_values("rating", ascending=False)
        res = res.reset_index(drop=True)

        
        # Average TransferMarket Value
        if self.avg_mv is not None:
            # apply box-cox transformation to normalize average market value, then standardize it to range (0,1)
            avg_mv = self.avg_mv
            avg_mv = avg_mv.copy()
            transformed_mv, _ = stats.boxcox(avg_mv['avg_market_val'])
            #transformed_mv = transformed_mv.copy()
            avg_mv.loc[:, 'transformed_mv'] = transformed_mv
            avg_mv.loc[:, 'transformed_mv'] = (avg_mv['transformed_mv'] - avg_mv['transformed_mv'].min()) / (avg_mv['transformed_mv'].max() - avg_mv['transformed_mv'].min())
            res = res.merge(avg_mv, left_on='team',right_on='team_id')
            res['mv_rating'] = res['rating'] + res['transformed_mv']
            columns_to_keep = ['team', 'rating', 'mv_rating', 'offence', 'defence', 'season']
            res = res[columns_to_keep]

        return res, home_advantage

# in matrix M, 1 = game played at home, -1 = game played away
def _build_m(fixtures, teams):
    n_teams = len(teams)
    M = np.zeros([n_teams+1, n_teams+1])

    # iterate over rows (games)
    for _, row in fixtures.iterrows():
        # return the index of current game's home team
        h = np.where(teams == row["team_home"])[0][0]
        # return the index of current game's away team
        a = np.where(teams == row["team_away"])[0][0]

        M[h, a] = M[h, a] - 1 * row['weighted_time']
        M[a, h] = M[a, h] - 1 * row['weighted_time']    

    # for each team, get the row sum (total games played) as disgonal entry
    for i in range(len(M)-1):
        team = teams[i]
        M[i, i] = np.abs(
            np.sum(
                M[i,]
            )
        )
        # add homefield advantage counted toward each team
        n_home = np.sum(team == fixtures['team_home'])
        n_away = np.sum(team == fixtures['team_away'])
        M[i, n_teams] = (n_home - n_away) 
    
    # fill in the last row
    M[-1,:] = M[:,-1]
    M[-1,-1] = len(fixtures)


    #new_row = np.hstack((np.ones(n_teams), 0))
    new_row = np.append(np.ones(n_teams), 0)
    M = np.vstack((M, new_row))
    
    return M


def _build_p(fixtures, teams):
    p = list()
    for team in teams:
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")

        # sum of all games' scores (home & away)
        goals_for = home["wtd_goals_home"].sum() + away["wtd_goals_away"].sum() 
        # sum of all game opponents' scores
        goals_away = home["wtd_goals_away"].sum() + away["wtd_goals_home"].sum() 

        p.append(goals_for - goals_away)
    
    # append the homefiled advantage term
    sum = np.sum(np.abs(fixtures['wtd_goals_home'] - fixtures['wtd_goals_away']))
    p.append(sum)   

    p.append(0)

    return p


def _solve_ratings(M, p):
    ratings = np.linalg.lstsq(M, p, rcond=None)[0]
    return ratings


def _solve_d(t, Tr_f):
    ratings = np.linalg.lstsq(t, Tr_f, rcond=None)[0]
    return ratings


def _build_t(fixtures, teams):
    n_teams = len(teams)
    t = np.zeros([n_teams+1, n_teams+1])

    for _, row in fixtures.iterrows():
        h = np.where(teams == row["team_home"])[0][0]
        a = np.where(teams == row["team_away"])[0][0]

        t[h, a] = t[h, a] + 1 * row['weighted_time']
        t[a, h] = t[a, h] + 1 * row['weighted_time']

    for i in range(len(t)-1):
        team = teams[i]
        t[i, i] = np.sum(
            t[i, ]
        )

        # add homefield advantage counted toward each team
        n_home = np.sum(team == fixtures['team_home'])
        n_away = np.sum(team == fixtures['team_away'])
        t[i, n_teams] = (n_home - n_away) 
    
    # fill in the last row
    t[-1,:] = t[:,-1]
    t[-1,-1] = len(fixtures)

    return t


def _build_f(fixtures, teams):
    f = list()
    for team in teams:
        home = fixtures.query("team_home == @team")
        away = fixtures.query("team_away == @team")
        goals_for = home["wtd_goals_home"].sum() + away["wtd_goals_away"].sum() 
        f.append(goals_for)
    f.append(0)
    return f
