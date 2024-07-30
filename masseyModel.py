from functools import cache
from time import perf_counter

import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from baseModel import BaseModel
from getData import fetch_data_for_massey_eos_eval
from weighted_massey_engine import WeightedMasseyEngine


class WeightedMasseyModel(BaseModel):
    _plot_title = "Weighted Massey Brier Score by Season and League"
    _plot_filename = "wtd_massey_brier_scores.png"

    @classmethod
    @cache
    def _get_model_by_season(cls, season):
        """ 
        Function returns a dataframe which contains the game statistics of given season and league (if input includes a valid league id, otherwise it returns stats of all games in that season)
            - one row of dataframe represents 1 game
            - each row also contains 
                1) massey ratings of home and away team, which are calculated using data from all previous seasons
                2) rating differentials = home team rating - away team rating
        """
        Games, ranking, marketValues = fetch_data_for_massey_eos_eval()

        pre_Games = Games.loc[(Games['season'] < season), :]
        #pre_Games = Games.loc[(Games['season'] < season) & (Games['league_id'].isin(league_ids)), :]

        avg_mv = marketValues.loc[marketValues['season'] == season, :]
    
        # get WEIGHTED massey ratings using data before predict season
        wm_ratings, wm_home_advantage = WeightedMasseyEngine.get_ratings(goals_home=pre_Games['fulltime_home_goals'], 
                                    goals_away=pre_Games['fulltime_away_goals'], teams_home=pre_Games['home_team_id'],
                                    teams_away=pre_Games['away_team_id'], match_date=pre_Games['date'], avg_mv=avg_mv)

        # if uses massey rating in combination with market value
        wm_ratings = wm_ratings.drop(columns=['rating'])
        wm_ratings = wm_ratings.rename(columns={'mv_rating':'rating'})

        # merge ratings with game data before predict season
        ratings = wm_ratings[['team', 'rating']].copy()

        home_rating = ratings.rename(columns={'team': 'home_team_id', 
                                            'rating': 'home_rating'})
        away_rating = ratings.rename(columns={'team': 'away_team_id', 
                                            'rating': 'away_rating'})
        Year_Games = Games.loc[Games['season']==season, :]
        Year_Games = pd.merge(Year_Games, home_rating, on=['home_team_id'], how='left')
        Year_Games = pd.merge(Year_Games, away_rating, on=['away_team_id'], how='left')
        Year_Games = Year_Games.dropna()

        Year_Games['home_rating'] = Year_Games['home_rating'] + wm_home_advantage

        Year_Games['rating_diff'] = Year_Games['home_rating'] - Year_Games['away_rating']

        return Year_Games

    @classmethod
    def getModel(cls, season, league):
        data = cls._get_model_by_season(season)
        if league is not None:
            data = data.loc[data['league_id'] == league, :]
        return data.copy()
    
    @classmethod
    def getPredProb(cls, season, league):
        """
        Get ordered probit model using game data before predict season
            y(win/draw/loss) = beta * (r_h - r_a)

        Return a dataframe containing 
            - game statistics of predict season
            - predicted probability of win, draw, loss of each game in predict season 
        """

        # change 2011 to (earliest season + 1) if dataset is updated
        # Concatenate all collected DataFrames into a single DataFrame
        pre_data = pd.concat([cls.getModel(Year, None) for Year in range(2011, season)], ignore_index=True)
        model = OrderedModel(pre_data['result'],pre_data['rating_diff'],distr = 'probit')
        model = model.fit(method='bfgs')

        # get games data and massey ratings for predict season and league
        pred_data = cls.getModel(season, league)

        predictions = [model.predict(i)[0] for i in pred_data["rating_diff"]]
        # Careful, the model.predict should return in the order of loss, draw, win...
        pred_data[["pred-loss","pred-draw","pred-win"]] = predictions
        return pred_data
    
    @classmethod
    def getBrierScores(cls, season, league):
        pred_data = cls.getPredProb(season, league)
        pred_data = pred_data.set_index("match_id")
        return super()._calc_brier_scores(pred_data)
    
    @classmethod
    def plotBrierScores(cls, seasons=range(2012,2024), **kwargs):
        return super().plotBrierScores(seasons=seasons, **kwargs)

class MasseyModel(WeightedMasseyModel):
    _plot_title = "Massey Brier Score by Season and League"
    _plot_filename = "massey_brier_scores.png"

    @classmethod
    @cache
    def _get_model_by_season(cls, season):
        """ 
        Function returns a dataframe which contains the game statistics of given season and league (if input includes a value league id, otherwise it returns stats of all games in that season)
            - one row of dataframe represents 1 game
            - each row also contains 
                1) massey ratings of home and away team, which are calculated using data from all previous seasons
                2) rating differentials = home team rating - away team rating
        """
        Games, ranking, marketValues = fetch_data_for_massey_eos_eval()

        pre_Games = Games.loc[Games['season'] < season, :]

        # get Massey ratings using 
        massey_ratings, home_advantage = WeightedMasseyEngine.get_ratings(goals_home=pre_Games['fulltime_home_goals'], 
                        goals_away=pre_Games['fulltime_away_goals'], teams_home=pre_Games['home_team_id'],
                        teams_away=pre_Games['away_team_id'])

        # merge ratings with game data
        ratings = massey_ratings[['team', 'rating']].copy()
        ratings.loc[:, 'season'] = season

        home_rating = ratings.rename(columns={'team': 'home_team_id', 
                                            'rating': 'home_rating'})
        away_rating = ratings.rename(columns={'team': 'away_team_id', 
                                            'rating': 'away_rating'})
        Year_Games = Games.loc[Games['season']==season, :]
        Year_Games = pd.merge(Year_Games, home_rating, on=['home_team_id', 'season'], how='left')
        Year_Games = pd.merge(Year_Games, away_rating, on=['away_team_id', 'season'], how='left')
        Year_Games = Year_Games.dropna()

        Year_Games['rating_diff'] = Year_Games['home_rating'] - Year_Games['away_rating']

        return Year_Games


if __name__ == "__main__":
    start = perf_counter()
    MasseyModel.plotBrierScores(country="england")
    WeightedMasseyModel.plotBrierScores(country="england")
    end = perf_counter()
    print(end-start)
